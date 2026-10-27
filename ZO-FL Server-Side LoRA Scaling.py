#!/usr/bin/env python3
import argparse
import contextlib
import json

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

TASKS = {
    "sst2":   ("{sentence} It was", [" terrible", " great"]),
    "cr":     ("{sentence} It was", [" terrible", " great"]),
    "sst5":   ("{sentence} It was", [" terrible", " bad", " okay", " good", " great"]),
    "subj":   ("{sentence} This is", [" objective", " subjective"]),
    "trec":   ("Question: {question}\nType:",
               [" Expression", " Entity", " Description", " Human", " Location", " Number"]),
    "agnews": ("{text}\nTopic:", [" World", " Sports", " Business", " Technology"]),
}


def load_data(path, template, n_classes):
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if not recs:
        raise SystemExit(f"{path}: no examples found")
    prompts = [template.format(**{k: v.strip() if isinstance(v, str) else v for k, v in r.items()})
               for r in recs]
    y = np.array([int(r["label"]) for r in recs])
    if y.min() < 0 or y.max() >= n_classes:
        raise SystemExit(f"{path}: labels must be in 0..{n_classes - 1}")
    return prompts, y


QUANT_LINEAR = ("QuantLinear", "HQQLinear")


@torch.no_grad()
def hqq_quantize(model, bits, group_size, device):
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    qc = BaseQuantizeConfig(nbits=bits, group_size=group_size)
    qc["weight_quant_params"]["axis"] = 1
    targets = [(parent, name) for full, parent in model.named_modules()
               for name, child in parent.named_children()
               if isinstance(child, torch.nn.Linear) and "layers." in f"{full}.{name}"]
    if not targets:
        raise SystemExit("No decoder linear layers found to quantize with HQQ")
    for parent, name in targets:
        lin = getattr(parent, name).to(device)
        setattr(parent, name, HQQLinear(lin, qc, compute_dtype=torch.float16, device=device, del_orig=True))


def load_base(model_path, quant, device, gptq_dir="", bits=4, group_size=32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if quant != "fp16" and not device.startswith("cuda"):
        raise SystemExit(f"--quant {quant} requires a GPU")
    if quant == "gptq":
        if not gptq_dir:
            raise SystemExit("--quant gptq requires --gptq-dir")
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            gptq_dir, device=device, inject_fused_attention=False, inject_fused_mlp=False).model
    else:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation="eager")
        if quant == "hqq":
            hqq_quantize(model, bits, group_size, device)
        model = model.to(device)
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


class ScaledLoRA(torch.nn.Module):
    def __init__(self, base, A, B, scaling, owner):
        super().__init__()
        self.base, self.owner, self.scaling = base, owner, scaling
        self.register_buffer("A", A)
        self.register_buffer("B", B)

    def forward(self, x, *args, **kwargs):
        out = self.base(x, *args, **kwargs)
        delta = F.linear(F.linear(x.to(self.A.dtype), self.A), self.B) * (self.scaling * self.owner.s)
        return out + delta.to(out.dtype)


def attach_adapter(model, path, alpha=None):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    layers = sorted(k[:-len(".lora_A")] for k in sd if k.endswith(".lora_A"))
    if not layers or any(n + ".lora_B" not in sd for n in layers):
        raise SystemExit(f"{path}: no lora_A/lora_B pairs found")
    dev = next(model.parameters()).device
    owner = argparse.Namespace(s=1.0)
    for name in layers:
        A, B = sd[name + ".lora_A"], sd[name + ".lora_B"]
        if A.shape[0] != B.shape[1]:
            raise SystemExit(f"{name}: rank mismatch between A {tuple(A.shape)} and B {tuple(B.shape)}")
        r = A.shape[0]
        parent_name, leaf = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        scaling = (r if alpha is None else alpha) / r
        setattr(parent, leaf, ScaledLoRA(getattr(parent, leaf), A.to(dev), B.to(dev), scaling, owner))
    return owner


def _pad(rows, pad_id, device):
    n = max(len(r) for r in rows)
    ids = torch.tensor([r + [pad_id] * (n - len(r)) for r in rows], device=device)
    mask = torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in rows], device=device)
    return ids, mask


def _logp_at(model, ids, mask, rows_pos):
    dec = model.get_decoder() if hasattr(model, "get_decoder") else None
    if dec is not None:
        h, head = dec(input_ids=ids, attention_mask=mask)[0], model.get_output_embeddings()
        return [F.log_softmax(head(h[i, pos]).float(), dim=-1) for i, pos in enumerate(rows_pos)]
    logits = model(input_ids=ids, attention_mask=mask).logits
    return [F.log_softmax(logits[i, pos].float(), dim=-1) for i, pos in enumerate(rows_pos)]


def _class_scores(model, tok, prompts, word_ids, max_len):
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    budget = max_len - max(len(w) for w in word_ids)
    heads = [tok(p).input_ids[-budget:] for p in prompts]
    if all(len(w) == 1 for w in word_ids):
        ids, mask = _pad(heads, pad_id, device)
        logp = _logp_at(model, ids, mask, [[len(h) - 1] for h in heads])
        return torch.cat(logp)[:, torch.tensor([w[0] for w in word_ids], device=device)]
    rows = [h + w for h in heads for w in word_ids]
    pos = [list(range(len(h) - 1, len(h) - 1 + len(w))) for h in heads for w in word_ids]
    ids, mask = _pad(rows, pad_id, device)
    logp = _logp_at(model, ids, mask, pos)
    tgt = [w for _ in heads for w in word_ids]
    out = [lp[torch.arange(len(w), device=device), torch.tensor(w, device=device)].sum()
           for lp, w in zip(logp, tgt)]
    return torch.stack(out).view(len(prompts), len(word_ids))


def _match_tangent_dtype(mod, inp, out):
    p, t = fwAD.unpack_dual(out)
    if t is not None and t.dtype != p.dtype:
        return fwAD.make_dual(p, t.to(p.dtype))
    return None


@contextlib.contextmanager
def _jvp_ready(model):
    undo = []
    for m in model.modules():
        if type(m).__name__ in QUANT_LINEAR:
            def fwd(x, *args, _orig=m.forward, _m=m, **kwargs):
                p, t = fwAD.unpack_dual(x)
                if t is None:
                    return _orig(x, *args, **kwargs)
                y = _orig(p, *args, **kwargs)
                ty = _orig(t.to(p.dtype), *args, **kwargs)
                if getattr(_m, "bias", None) is not None:
                    ty = ty - _m.bias.to(ty.dtype)
                return fwAD.make_dual(y, ty.to(y.dtype))
            m.forward = fwd
            undo.append(lambda m=m: delattr(m, "forward"))
        elif isinstance(m, torch.nn.LayerNorm):
            undo.append(m.register_forward_hook(_match_tangent_dtype).remove)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attn = sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        attn = contextlib.nullcontext()
    try:
        with attn:
            yield
    finally:
        for u in undo:
            u()


def jvp_scores(model, tok, owner, prompts, word_ids, max_len, bs):
    device = next(model.parameters()).device
    Z, T = [], []
    with torch.no_grad(), _jvp_ready(model), fwAD.dual_level():
        owner.s = fwAD.make_dual(torch.tensor(1.0, device=device), torch.tensor(1.0, device=device))
        try:
            for i in range(0, len(prompts), bs):
                p, t = fwAD.unpack_dual(_class_scores(model, tok, prompts[i:i + bs], word_ids, max_len))
                if t is None:
                    raise RuntimeError("Tangent did not propagate to the class scores")
                Z.append(p.double().cpu().numpy())
                T.append(t.double().cpu().numpy())
        finally:
            owner.s = 1.0
    return np.concatenate(Z), np.concatenate(T)


def scaling_factor(z1, zdot, y):
    if not (np.isfinite(z1).all() and np.isfinite(zdot).all()):
        raise RuntimeError("Non-finite values in the class scores or their derivatives")
    a, d, idx = z1 - zdot, zdot, np.arange(len(y))

    def grad(s):
        Z = a + s * d
        P = np.exp(Z - Z.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        return float(((P * d).sum(axis=1) - d[idx, y]).mean())

    if not (grad(0.0) < 0 < float((d.max(axis=1) - d[idx, y]).mean())):
        return 1.0
    lo, hi = 0.0, 1.0
    while grad(hi) < 0:
        lo, hi = hi, 2 * hi
    while hi - lo > 1e-12 * hi:
        mid = 0.5 * (lo + hi)
        if grad(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Server-side LoRA scaling: estimate the scaling factor with a single JVP")
    ap.add_argument("--model", required=True, help="path to the fp16 model (also used for the tokenizer)")
    ap.add_argument("--quant", choices=["fp16", "gptq", "hqq"], default="fp16", help="base model type")
    ap.add_argument("--gptq-dir", default="", help="path to the GPTQ model saved by auto-gptq")
    ap.add_argument("--bits", type=int, choices=[2, 3, 4, 8], default=4, help="HQQ bit width")
    ap.add_argument("--group-size", type=int, default=32, help="HQQ group size")
    ap.add_argument("--adapter", required=True, help="aggregated (FedAvg) adapter .pt with lora_A/lora_B")
    ap.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha (default: r)")
    ap.add_argument("--data", required=True, help="server validation data (.jsonl)")
    ap.add_argument("--task", choices=sorted(TASKS), default="", help="built-in task")
    ap.add_argument("--template", default="", help="prompt template used instead of --task")
    ap.add_argument("--label-words", nargs="+", default=None, help="label words used instead of --task")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--device", default="")
    a = ap.parse_args(argv)

    template, words = TASKS[a.task] if a.task else (a.template, a.label_words)
    if not template or not words:
        ap.error("specify --task, or both --template and --label-words")
    if a.batch_size < 1 or a.group_size < 1:
        ap.error("--batch-size and --group-size must be at least 1")
    device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    prompts, y = load_data(a.data, template, len(words))
    tok, model = load_base(a.model, "gptq" if a.gptq_dir and a.quant == "fp16" else a.quant,
                           device, a.gptq_dir, a.bits, a.group_size)
    word_ids = [tok(w, add_special_tokens=False).input_ids for w in words]
    if any(not w for w in word_ids) or a.max_len <= max(len(w) for w in word_ids):
        ap.error("every label word must produce at least one token, and --max-len must exceed the longest label")
    owner = attach_adapter(model, a.adapter, a.lora_alpha)

    z1, zdot = jvp_scores(model, tok, owner, prompts, word_ids, a.max_len, a.batch_size)
    print(f"s_hat = {scaling_factor(z1, zdot, y):.6g}")


if __name__ == "__main__":
    main()
