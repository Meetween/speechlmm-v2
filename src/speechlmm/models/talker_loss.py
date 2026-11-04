"""Shared Talker teacher-forcing helpers (pre-shifted CE, streaming text, dummy graph)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_streaming_text(trailing: torch.Tensor, pad_slot: torch.Tensor, length: int) -> torch.Tensor:
    """Align trailing text to `length`, then hold `pad_slot`."""
    if trailing.dim() == 3:
        trailing = trailing.reshape(trailing.shape[1], trailing.shape[2])
    pad_slot = pad_slot.reshape(-1, pad_slot.shape[-1])[:1]
    n = trailing.shape[0]
    if n < length:
        return torch.cat([trailing, pad_slot.expand(length - n, -1)], dim=0)
    return trailing[:length]


def codec_targets(
    context_len: int,
    prefill_len: int,
    codes: torch.Tensor,
    eos_id: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Pre-shifted targets: first valid at ``context_len + prefill_len - 1``."""
    if codes.dim() != 1:
        raise ValueError(f"codec_targets expects 1-D codes, got shape {tuple(codes.shape)}")
    ignore_len = context_len + prefill_len - 1
    target = torch.cat(
        [
            torch.full((ignore_len,), ignore_index, device=codes.device, dtype=codes.dtype),
            codes,
            codes.new_tensor([eos_id]),
        ]
    )
    if int((target != ignore_index).sum().item()) != codes.numel() + 1:
        raise AssertionError("Talker target alignment invariant failed")
    return target


def reduce_talker_losses(
    per_token_sums: list[torch.Tensor],
    valid_counts: list[int | torch.Tensor],
    reduction: str,
) -> torch.Tensor:
    if not per_token_sums:
        raise ValueError("reduce_talker_losses got no losses")
    if reduction == "sample":
        means = []
        for loss_sum, count in zip(per_token_sums, valid_counts):
            n = int(count.item()) if torch.is_tensor(count) else int(count)
            means.append(loss_sum / max(n, 1))
        return torch.stack(means).mean()
    total_sum = per_token_sums[0]
    for extra in per_token_sums[1:]:
        total_sum = total_sum + extra
    total_n = 0
    for count in valid_counts:
        total_n += int(count.item()) if torch.is_tensor(count) else int(count)
    if total_n == 0:
        return total_sum * 0
    return total_sum / total_n


def zero_grad_edge(*tensors: torch.Tensor | None) -> torch.Tensor | None:
    edge = None
    for tensor in tensors:
        if tensor is None:
            continue
        piece = tensor.reshape(-1)[:1].sum() * 0
        edge = piece if edge is None else edge + piece
    return edge


def valid_codec_codes(labels_row: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if labels_row.dim() == 1:
        return labels_row[labels_row != ignore_index]
    mask = (labels_row != ignore_index).any(dim=-1)
    return labels_row[mask]


def first_codebook(codes: torch.Tensor) -> torch.Tensor:
    if codes.dim() == 2:
        return codes[:, 0]
    return codes


def exclude_trailing_im_end(token_ids: torch.Tensor, end: int, im_end_id: int) -> int:
    if end > 0 and int(token_ids[end - 1].item()) == int(im_end_id):
        return end - 1
    return end


def rvq_codec_embeds(talker: torch.nn.Module, codes: torch.Tensor) -> torch.Tensor:
    """Embed codec frames. Book 0 from Talker; residual books from the predictor."""
    embed0 = talker.model.codec_embedding if hasattr(talker, "model") else talker.codec_embedding
    if codes.dim() == 1:
        return embed0(codes)
    hidden = embed0(codes[:, 0])
    predictor = getattr(talker, "code_predictor", None)
    residual_embeds = getattr(getattr(predictor, "model", None), "codec_embedding", None)
    if residual_embeds is None:
        return hidden
    groups = min(codes.size(-1) - 1, len(residual_embeds))
    for group in range(groups):
        hidden = hidden + residual_embeds[group](codes[:, group + 1])
    return hidden


def _frame_width(codec_tokens) -> int | None:
    if not codec_tokens:
        return None
    first = codec_tokens[0]
    if isinstance(first, (list, tuple)):
        return len(first)
    return 1


def pad_codec_batch(
    batch: list,
    num_codebooks: int | None = None,
    ignore_index: int = -100,
    dummy: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pad inline codec tokens to `[B, C]` or `[B, C, G]`.

    Returns ``(codec_labels, codec_mask)``. ``codec_mask[i]`` is True iff sample
    ``i`` had real codes. When *dummy* is True and the batch has no codes,
    emits a 1-frame zero tensor with all-False mask.
    """
    present = [ct for ct in batch if ct is not None and len(ct) > 0]
    if not present:
        if not dummy:
            return None, None
        bsz = max(len(batch), 1)
        groups = 1 if not num_codebooks or num_codebooks <= 1 else int(num_codebooks)
        if groups <= 1:
            labels = torch.zeros((bsz, 1), dtype=torch.long)
        else:
            labels = torch.zeros((bsz, 1, groups), dtype=torch.long)
        return labels, torch.zeros((bsz,), dtype=torch.bool)

    widths = {_frame_width(ct) for ct in present}
    if None in widths or len(widths) != 1:
        raise ValueError(f"Inconsistent codec_tokens layout in batch: {widths}")
    groups = widths.pop()
    if num_codebooks is not None and groups != int(num_codebooks):
        raise ValueError(
            f"codec_tokens has {groups} codebooks, config.talker_num_codebooks={num_codebooks}"
        )

    max_c = max(len(ct) for ct in present)
    mask = []
    if groups == 1:
        labels = []
        for ct in batch:
            if ct is None or len(ct) == 0:
                labels.append([ignore_index] * max_c)
                mask.append(False)
            else:
                if isinstance(ct[0], (list, tuple)):
                    flat = [int(frame[0]) for frame in ct]
                else:
                    flat = [int(x) for x in ct]
                labels.append(flat + [ignore_index] * (max_c - len(flat)))
                mask.append(True)
        return torch.tensor(labels, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    labels = []
    for ct in batch:
        if ct is None or len(ct) == 0:
            labels.append([[ignore_index] * groups for _ in range(max_c)])
            mask.append(False)
        else:
            row = [[int(x) for x in frame] for frame in ct]
            if any(len(frame) != groups for frame in row):
                raise ValueError("Inconsistent per-frame codebook width in codec_tokens")
            row.extend([[ignore_index] * groups] * (max_c - len(row)))
            labels.append(row)
            mask.append(True)
    return torch.tensor(labels, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def token_ce_sum(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> tuple[torch.Tensor, int]:
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    valid = int((target != ignore_index).sum().item())
    return loss, valid


def continue_position_ids(prefix_position_ids: torch.Tensor, extra_len: int) -> torch.Tensor:
    """Append sequential mRoPE ids after a `get_rope_index` prefix of shape `[3, B, T]`."""
    if extra_len <= 0:
        return prefix_position_ids
    last = prefix_position_ids[..., -1:]
    extra = torch.arange(1, extra_len + 1, device=prefix_position_ids.device, dtype=prefix_position_ids.dtype)
    extra = extra.view(1, 1, extra_len).expand(
        prefix_position_ids.size(0), prefix_position_ids.size(1), extra_len
    )
    return torch.cat([prefix_position_ids, last + extra], dim=-1)


def sequential_position_ids(batch: int, seq_len: int, device, dtype=torch.long) -> torch.Tensor:
    ids = torch.arange(seq_len, device=device, dtype=dtype).view(1, 1, seq_len).expand(3, batch, seq_len)
    return ids.clone()


def dummy_talker_forward_loss(talker: torch.nn.Module, device, dtype) -> torch.Tensor | None:
    """1-frame Talker forward with a zeroed graph edge (ZeRO-3 rank symmetry)."""
    params = [p for p in talker.parameters() if p.requires_grad]
    if not params:
        return None
    head = talker.codec_head
    model = talker.model
    proj = getattr(talker, "thinker_to_talker_proj", None)
    if proj is not None:
        hidden = proj.in_features
        embeds = torch.zeros(1, 1, hidden, device=device, dtype=dtype)
        embeds = proj(embeds)
    else:
        hidden = head.in_features
        embeds = torch.zeros(1, 1, hidden, device=device, dtype=dtype)
    attention_mask = torch.ones(1, 1, device=device, dtype=torch.long)
    hidden_out = model(inputs_embeds=embeds, attention_mask=attention_mask)[0]
    logits = head(hidden_out)
    loss = logits.float().sum() * 0
    predictor = getattr(talker, "code_predictor", None)
    if predictor is not None and any(p.requires_grad for p in predictor.parameters()):
        pred_h = getattr(getattr(predictor, "config", None), "hidden_size", logits.size(-1))
        pred_in = torch.zeros(1, 2, pred_h, device=device, dtype=dtype)
        pred_out = predictor.model(inputs_embeds=pred_in)[0]
        loss = loss + pred_out.float().sum() * 0
    return loss + params[0].reshape(-1)[:1].sum() * 0
