"""Exact, faster sampling for Kronos via a shared-prefix KV cache.

The reference `auto_regressive_inference` in the Kronos repo re-runs the full
context window through the transformer for every sampled path and every
forecast step. When lookback + pred_len - 1 <= max_context the window never
slides, so the context can be encoded ONCE per series, its keys/values cached,
and every sample path decoded one token at a time against that cache. The same
trick applies to the causal tokenizer decoder that maps tokens back to OHLCV.

The computation is the same as the reference implementation up to float
rounding (checked by tests/test_kronos_fast.py with greedy decoding).
"""

import torch
import torch.nn.functional as F


def _rope(x, pos, inv_freq):
    """Rotary embedding as in Kronos' RotaryPositionalEmbedding, at given positions."""
    freqs = torch.outer(pos.to(inv_freq.dtype), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _qkv(attn, h):
    n, t, _ = h.shape
    split = lambda z: z.view(n, t, attn.n_heads, attn.head_dim).transpose(1, 2)  # noqa: E731
    return split(attn.q_proj(h)), split(attn.k_proj(h)), split(attn.v_proj(h))


def _prefix_block(layer, x):
    """TransformerBlock forward over a full prefix; returns output and (K, V) cache."""
    attn = layer.self_attn
    q, k, v = _qkv(attn, layer.norm1(x))
    pos = torch.arange(x.shape[1], device=x.device)
    q, k = _rope(q, pos, attn.rotary.inv_freq), _rope(k, pos, attn.rotary.inv_freq)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    o = o.transpose(1, 2).reshape(x.shape)
    x = x + attn.out_proj(o)
    x = x + layer.ffn(layer.norm2(x))
    return x, (k, v)


def _cached_attention(q, kp, vp, kn, vn):
    """q (B,S,H,t,hd) attends to prefix (B,H,Lp,hd) and causally to new keys (B,S,H,Tn,hd)."""
    scale = q.shape[-1] ** -0.5
    sp = torch.einsum("bshtd,bhld->bshtl", q, kp) * scale
    sn = torch.einsum("bshtd,bshld->bshtl", q, kn) * scale
    t, tn = q.shape[-2], kn.shape[-2]
    allowed = torch.arange(tn, device=q.device)[None, :] <= (tn - t + torch.arange(t, device=q.device))[:, None]
    sn = sn.masked_fill(~allowed, float("-inf"))
    w = torch.softmax(torch.cat([sp, sn], dim=-1), dim=-1)
    lp = kp.shape[-2]
    return (torch.einsum("bshtl,bhld->bshtd", w[..., :lp], vp)
            + torch.einsum("bshtl,bshld->bshtd", w[..., lp:], vn))


def _step_block(layer, x, start_pos, prefix_kv, new_kv, B, S):
    """TransformerBlock forward for new tokens x (B*S, t, d) given caches.

    new_kv holds this layer's keys/values of earlier new tokens (or None); the
    updated cache is returned.
    """
    attn = layer.self_attn
    n, t, d = x.shape
    q, k, v = _qkv(attn, layer.norm1(x))
    pos = start_pos + torch.arange(t, device=x.device)
    q, k = _rope(q, pos, attn.rotary.inv_freq), _rope(k, pos, attn.rotary.inv_freq)
    if new_kv is not None:
        k = torch.cat([new_kv[0], k], dim=2)
        v = torch.cat([new_kv[1], v], dim=2)
    sh = lambda z: z.view(B, S, *z.shape[1:])  # noqa: E731
    o = _cached_attention(sh(q), prefix_kv[0], prefix_kv[1], sh(k), sh(v))
    o = o.reshape(n, attn.n_heads, t, attn.head_dim).transpose(1, 2).reshape(n, t, d)
    x = x + attn.out_proj(o)
    x = x + layer.ffn(layer.norm2(x))
    return x, (k, v)


def _cross_kv(cross, h):
    n, t, _ = h.shape
    split = lambda z: z.view(n, t, cross.n_heads, cross.head_dim).transpose(1, 2)  # noqa: E731
    return split(cross.k_proj(h)), split(cross.v_proj(h))


@torch.no_grad()
def sample_paths(tokenizer, model, x, x_stamp, y_stamp, pred_len, samples,
                 max_context=512, T=1.0, top_k=0, top_p=0.9, sampler=None):
    """Sample `samples` future paths per series.

    x: (B, L, 6) normalised + clipped features; x_stamp: (B, L, 5);
    y_stamp: (B, pred_len, 5). Returns (B, samples, pred_len, 6) in the
    normalised space (same as the reference before de-normalisation).
    """
    if sampler is None:
        from model.kronos import sample_from_logits as sampler
    B, L, _ = x.shape
    S = samples
    N = B * S
    assert L + pred_len - 1 <= max_context, "window would slide; reduce lookback"

    # ---- context: tokenize + one transformer pass per series
    s1, s2 = tokenizer.encode(x, half=True)
    h = model.embedding([s1, s2]) + model.time_emb(x_stamp)
    prefix_kv = []
    for layer in model.transformer:
        h, kv = _prefix_block(layer, h)
        prefix_kv.append(kv)
    hidden_p = model.norm(h)                                   # (B, L, d)
    cross = model.dep_layer.cross_attn
    ck_p, cv_p = _cross_kv(cross, hidden_p)                    # (B, Hc, L, hd)

    rep = lambda z: z.repeat_interleave(S, dim=0)              # noqa: E731
    y_stamp_n = rep(y_stamp)
    new_kv = [None] * len(model.transformer)
    ck_n = cv_n = None
    h_last = rep(hidden_p[:, -1:])                             # (N, 1, d)
    gen1, gen2 = [], []
    for i in range(pred_len):
        if i > 0:
            e = model.embedding([gen1[-1], gen2[-1]]) + model.time_emb(y_stamp_n[:, i - 1: i])
            for j, layer in enumerate(model.transformer):
                e, new_kv[j] = _step_block(layer, e, L + i - 1, prefix_kv[j], new_kv[j], B, S)
            h_last = model.norm(e)
            k, v = _cross_kv(cross, h_last)
            ck_n = k if ck_n is None else torch.cat([ck_n, k], dim=2)
            cv_n = v if cv_n is None else torch.cat([cv_n, v], dim=2)

        t1 = sampler(model.head(h_last[:, -1]), temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
        # s2 | s1: cross-attention from the s1 embedding to all context states.
        qe = model.embedding.emb_s1(t1)                        # (N, 1, d)
        q = cross.q_proj(qe).view(N, 1, cross.n_heads, cross.head_dim).transpose(1, 2)
        sh = lambda z: z.view(B, S, *z.shape[1:])              # noqa: E731
        if ck_n is None:
            kn = torch.zeros(N, cross.n_heads, 0, cross.head_dim, dtype=q.dtype, device=q.device)
            vn = kn
        else:
            kn, vn = ck_n, cv_n
        scale = cross.head_dim ** -0.5
        sp = torch.einsum("bshtd,bhld->bshtl", sh(q), ck_p) * scale
        sn = torch.einsum("bshtd,bshld->bshtl", sh(q), sh(kn)) * scale
        w = torch.softmax(torch.cat([sp, sn], -1), -1)
        o = (torch.einsum("bshtl,bhld->bshtd", w[..., :L], cv_p)
             + torch.einsum("bshtl,bshld->bshtd", w[..., L:], sh(vn)))
        o = o.reshape(N, cross.n_heads, 1, cross.head_dim).transpose(1, 2).reshape(N, 1, -1)
        x2 = model.dep_layer.norm(h_last[:, -1:] + cross.out_proj(o))
        t2 = sampler(model.head.cond_forward(x2)[:, -1], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
        gen1.append(t1.view(N, 1))
        gen2.append(t2.view(N, 1))

    # ---- detokenize: causal decoder over the last `max_context` tokens
    start = max(0, L + pred_len - max_context)
    zp = tokenizer.post_quant_embed(tokenizer.indices_to_bits([s1[:, start:], s2[:, start:]], half=True))
    dec_kv = []
    for layer in tokenizer.decoder:
        zp, kv = _prefix_block(layer, zp)
        dec_kv.append(kv)
    g1, g2 = torch.cat(gen1, 1), torch.cat(gen2, 1)
    zn = tokenizer.post_quant_embed(tokenizer.indices_to_bits([g1, g2], half=True))
    for j, layer in enumerate(tokenizer.decoder):
        zn, _ = _step_block(layer, zn, L - start, dec_kv[j], None, B, S)
    z = tokenizer.head(zn)                                     # (N, pred_len, 6)
    return z.view(B, S, pred_len, -1)
