"""Independent CPU DFlash2 forward on captured hidden vectors; temporary arrays stay outside receipts."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def rounded(x):
    return x.to(torch.bfloat16)


def away(x):
    return x.sign() * (x.abs() + 0.5).floor()


def quantized(w):
    shape = w.shape
    g = w.float().reshape(-1, 64)
    lo = g.min(-1).values
    hi = torch.maximum(g.max(-1).values, torch.zeros_like(lo))
    bias = torch.where(lo.abs() > hi.abs(), lo, hi)
    scale = ((hi - lo) / 15).clamp_min(1e-7)
    scale = torch.where(bias >= 0, -scale, scale)
    zero = away(bias / scale)
    scale = torch.where(zero != 0, bias / zero, scale)
    bias = torch.where(zero != 0, bias, 0)
    codes = away((g - bias[:, None]) / scale[:, None]).clamp(0, 15)
    return rounded(codes * rounded(scale).float()[:, None] + rounded(bias).float()[:, None]).reshape(shape)


class Reference:
    def __init__(self, draft, model, mode):
        self.c = json.loads((draft / "config.json").read_text())
        self.source = safe_open(draft / "model.safetensors", framework="pt", device="cpu")
        self.mode = mode
        self.model = model

    def weight(self, name):
        w = self.source.get_tensor(name)
        return quantized(w) if self.mode == "affine4" and w.ndim == 2 else w

    def linear(self, x, name):
        return rounded(x.float() @ self.weight(name).float().T)

    def norm(self, x, name):
        inv = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.c["rms_norm_eps"])
        return rounded(rounded(x.float() * inv).float() * self.weight(name).float())

    def conv(self, x, dynamic, name, branch, residual=None):
        d = self.c["hidden_size"]
        group = self.c["dflash_config"]["conv_group_size"]
        base = self.weight(name)[branch]
        dynamic = dynamic.reshape(x.shape[0], 2, 2, d // group)[:, branch]
        total = torch.zeros_like(x)
        for lag in range(2):
            values = x if lag == 0 else torch.cat([torch.zeros_like(x[:1]), x[:-1]])
            total = rounded(total.float() + rounded(base[lag].float() * values.float()).float())
            coeff = dynamic[:, lag].repeat_interleave(group, -1)
            total = rounded(total.float() + rounded(coeff.float() * values.float()).float())
        return rounded(total.float() + residual.float()) if residual is not None else total

    def rope(self, x, positions, heads):
        v = x.reshape(-1, heads, 128).float()
        angles = positions.double()[:, None] / self.c["rope_parameters"]["rope_theta"] ** (torch.arange(64).double() / 64)
        cs, sn = angles.cos().float()[:, None], angles.sin().float()[:, None]
        first, last = v[..., :64], v[..., 64:]
        return rounded(torch.cat([first * cs - last * sn, last * cs + first * sn], -1))

    def embed(self, ids):
        index = self.model / "model.safetensors.index.json"
        if index.exists():
            files = [self.model / json.loads(index.read_text())["weight_map"]["model.embed_tokens.weight"]]
        else:
            files = list(self.model.glob("*.safetensors"))
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as source:
                if "model.embed_tokens.weight" in source.keys():
                    return source.get_tensor("model.embed_tokens.weight")[ids]
        raise ValueError("missing shared target embedding")

    def forward(self, taps):
        context = self.norm(self.linear(taps, "fc.weight"), "hidden_norm.weight")
        x = self.embed([42] + [self.c["dflash_config"]["mask_token_id"]] * 7)
        for layer in range(self.c["num_hidden_layers"]):
            p = f"layers.{layer}."
            a = p + "self_attn."
            residual = x
            xn = self.norm(x, p + "input_layernorm.weight")
            dynamic = self.linear(xn, p + "attention_conv.kernel_projection.weight")
            xi = self.conv(xn, dynamic, p + "attention_conv.base_kernel", 0)
            q = self.linear(xi, a + "q_proj.weight").reshape(8, 32, 128)
            q = self.rope(self.norm(q, a + "q_norm.weight"), torch.arange(1, 9), 32)
            keys, values = [], []
            for h, positions in [(context, torch.tensor([0])), (xi, torch.arange(1, 9))]:
                k = self.linear(h, a + "k_proj.weight").reshape(-1, 8, 128)
                keys.append(self.rope(self.norm(k, a + "k_norm.weight"), positions, 8))
                values.append(self.linear(h, a + "v_proj.weight").reshape(-1, 8, 128))
            k = torch.cat(keys).repeat_interleave(4, 1).permute(1, 0, 2).float()
            v = torch.cat(values).repeat_interleave(4, 1).permute(1, 0, 2).float()
            probs = torch.softmax((q.permute(1, 0, 2).float() @ k.transpose(1, 2)) / 128**0.5, -1)
            attended = rounded(probs @ v).permute(1, 0, 2).reshape(8, 4096)
            branch = self.linear(attended, a + "o_proj.weight")
            x = self.conv(branch, dynamic, p + "attention_conv.base_kernel", 1, residual)
            residual = x
            xn = self.norm(x, p + "post_attention_layernorm.weight")
            dynamic = self.linear(xn, p + "mlp_conv.kernel_projection.weight")
            xi = self.conv(xn, dynamic, p + "mlp_conv.base_kernel", 0)
            gate = self.linear(xi, p + "mlp.gate_proj.weight")
            up = self.linear(xi, p + "mlp.up_proj.weight")
            exp = rounded(torch.exp(-gate.float()))
            den = rounded(1 + exp.float())
            sigmoid = rounded(1 / den.float())
            silu = rounded(gate.float() * sigmoid.float())
            branch = self.linear(rounded(silu.float() * up.float()), p + "mlp.down_proj.weight")
            x = self.conv(branch, dynamic, p + "mlp_conv.base_kernel", 1, residual)
        return self.norm(x[1:], "norm.weight")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--compare", type=Path)
    p.add_argument("--max-relative-l2", type=float, default=0.03)
    p.add_argument("--format", choices=["bf16", "affine4"], default="bf16")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    args.work.mkdir(parents=True, exist_ok=True)
    layers = [args.captures / f"engine-12-s0-layer{i}.f32" for i in range(5)]
    x = torch.cat([torch.from_numpy(np.fromfile(path, dtype=np.float32)) for path in layers])
    taps = rounded(x).reshape(1, -1)
    (args.work / "taps.bf16").write_bytes(taps.view(torch.uint16).numpy().tobytes())
    print(json.dumps({"input_layers": [0, 1, 2, 3, 4], "trained_taps": [4, 15, 26, 38, 49],
                      "note": "real captured vectors used as diagnostic inputs; no acceptance credit"}), flush=True)
    if args.compare:
        want = Reference(args.draft, args.model, args.format).forward(taps).float()
        got = torch.from_numpy(np.fromfile(args.compare, dtype=np.uint16).copy()).view(torch.bfloat16).reshape_as(want).float()
        delta = got - want
        print(json.dumps({"format": args.format, "elements": got.numel(), "finite": bool(got.isfinite().all()),
                          "relative_l2": float(delta.norm() / want.norm()), "max_abs": float(delta.abs().max()),
                          "rmse": float(delta.square().mean().sqrt())}), flush=True)
        if not got.isfinite().all() or float(delta.norm() / want.norm()) > args.max_relative_l2:
            raise SystemExit("native drafter missed the independent reference bound")


if __name__ == "__main__":
    main()
