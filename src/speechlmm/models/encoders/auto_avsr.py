# ============================================================================
# VENDORED FILE — DO NOT EDIT
# ============================================================================
# This is a self-contained copy of the AutoAVSR encoder bundled with its
# espnet dependencies so that it can run standalone without installing espnet.
#
# Source: https://github.com/mpc001/auto_avsr
# Pinned to commit 182b628:
#   https://github.com/mpc001/auto_avsr/commit/182b62837773ab01052d4ac21ef1d2203ea7d267
#
# TODO: Replace with a proper pip dependency or git submodule once auto_avsr
#       publishes a stable package.  Any local patches MUST be documented below.
#
# Local patches applied on top of upstream:
#   (none)
# ============================================================================
#
# Pre-trained model links at time of writing:
# | Model                                 | Training data (h)  |  WER [%]   |  Params (M) |    MD5            |
# |---------------------------------------|:------------------:|:----------:|:-----------:|:------------------------:|
# | [`vsr_trlrs3_23h_base.pth`](https://drive.google.com/file/d/1FDDTOBteJV8yBiJ8yePtZ-C-xR4s80rV/view?usp=sharing)             |        438          |    93.0    |     250     | fc8db  |
# | [`vsr_trlrs3_base.pth`](https://drive.google.com/file/d/12PNM5szUsk_CuaV1yB9dL_YWvSM1zvAd/view?usp=sharing)                 |        438          |    36.0    |     250     | c00a7  |
# | [`vsr_trlrs3vox2_base.pth`](https://drive.google.com/file/d/1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i/view?usp=sharing)             |        1759         |    24.6    |     250     | 774a6  |
# | [`vsr_trlrs2lrs3vox2avsp_base.pth`](https://drive.google.com/file/d/1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_/view?usp=sharing)     |        3291         |    20.3    |     250     | 49f77  |
# | [`asr_trlrs3_base.pth`](https://drive.google.com/file/d/1IBMkI7XyZo8mF3rz109rXrMH7MyxRuiY/view?usp=sharing)                 |        438          |    2.0     |     243     | 8af72  |
# | [`asr_trlrs3vox2_base.pth`](https://drive.google.com/file/d/1YN9lwZN6iWn2qNQRpfpGpnf2r6ZTQqVT/view?usp=sharing)             |        1759         |    1.0     |     243     | f0c5c  |


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unility funcitons for Transformer."""

import copy
import logging
import math
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Scorer interface module."""


class ScorerInterface:
    """Scorer interface for beam search.

    The scorer performs scoring of the all tokens in vocabulary.

    Examples:
        * Search heuristics
            * :class:`espnet.nets.scorers.length_bonus.LengthBonus`
        * Decoder networks of the sequence-to-sequence models
            * :class:`espnet.nets.pytorch_backend.nets.transformer.decoder.Decoder`
            * :class:`espnet.nets.pytorch_backend.nets.rnn.decoders.Decoder`
        * Neural language models
            * :class:`espnet.nets.pytorch_backend.lm.transformer.TransformerLM`
            * :class:`espnet.nets.pytorch_backend.lm.default.DefaultRNNLM`
            * :class:`espnet.nets.pytorch_backend.lm.seq_rnn.SequentialRNNLM`

    """

    def init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        return None

    def select_state(self, state: Any, i: int, new_id: int = None) -> Any:
        """Select state with relative ids in the main beam search.

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label index to select a state if necessary

        Returns:
            state: pruned state

        """
        return None if state is None else state[i]

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys

        """
        raise NotImplementedError

    def final_score(self, state: Any) -> float:
        """Score eos (optional).

        Args:
            state: Scorer state for prefix tokens

        Returns:
            float: final score

        """
        return 0.0


class BatchScorerInterface(ScorerInterface):
    """Batch scorer interface."""

    def batch_init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        return self.init_state(x)

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        warnings.warn(
            "{} batch score is implemented through for loop not parallelized".format(
                self.__class__.__name__
            )
        )
        scores = list()
        outstates = list()
        for i, (y, state, x) in enumerate(zip(ys, states, xs)):
            score, outstate = self.score(y, state, x)
            outstates.append(outstate)
            scores.append(score)
        scores = torch.cat(scores, 0).view(ys.shape[0], -1)
        return scores, outstates


class PartialScorerInterface(ScorerInterface):
    """Partial scorer interface for beam search.

    The partial scorer performs scoring when non-partial scorer finished scoring,
    and receives pre-pruned next tokens to score because it is too heavy to score
    all the tokens.

    Examples:
         * Prefix search for connectionist-temporal-classification models
             * :class:`espnet.nets.scorers.ctc.CTCPrefixScorer`

    """

    def score_partial(
        self, y: torch.Tensor, next_tokens: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        raise NotImplementedError


class BatchPartialScorerInterface(BatchScorerInterface, PartialScorerInterface):
    """Batch partial scorer interface for beam search."""

    def batch_score_partial(
        self,
        ys: torch.Tensor,
        next_tokens: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            next_tokens (torch.Tensor): torch.int64 tokens to score (n_batch, n_token).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for ys that has a shape `(n_batch, n_vocab)`
                and next states for ys
        """
        raise NotImplementedError


def add_sos_eos(ys_pad, sos, eos, ignore_id):
    """Add <sos> and <eos> labels.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int sos: index of <sos>
    :param int eos: index of <eeos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    try:
        from espnet.nets.pytorch_backend.nets_utils import pad_list as _pad_list
    except ImportError:
        _pad_list = pad_list

    _sos = ys_pad.new([sos])
    _eos = ys_pad.new([eos])
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return _pad_list(ys_in, eos), _pad_list(ys_out, ignore_id)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask, rtn_attn=False):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
            rtn_attn (boolean): Flag of return attention score
        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        if rtn_attn:
            return self.linear_out(x), self.attn
        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, rtn_attn=False):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            rtn_attn (boolean): Flag of return attention score
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask, rtn_attn)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""


class ConvolutionModule(torch.nn.Module):
    def __init__(self, channels, kernel_size, bias=True):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_cov1 = torch.nn.Conv1d(channels, 2 * channels, 1, bias=bias)
        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = torch.nn.BatchNorm1d(channels)
        self.pointwise_cov2 = torch.nn.Conv1d(channels, channels, 1, bias=bias)
        self.activation = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.nn.functional.glu(self.pointwise_cov1(x), dim=1)
        x = self.activation(self.norm(self.depthwise_conv(x)))
        x = self.pointwise_cov2(x)
        return x.transpose(1, 2)


class EncoderLayer(torch.nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.
        MultiHeadedAttention self_attn: self attention module
        RelPositionMultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward:
        feed forward module
    :param espnet.nets.pytorch_backend.transformer.convolution.
        ConvolutionModule feed_foreard:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param bool macaron_style: whether to use macaron style for PositionwiseFeedForward

    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        macaron_style=False,
        positionwise_layer_args=(),
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.ff_scale = 1.0
        self.conv_module = conv_module
        self.macaron_style = macaron_style
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if self.macaron_style:
            self.feed_forward_macaron = PositionwiseFeedForward(*positionwise_layer_args)
            self.ff_scale = 0.5
            # for another FNN module in macaron style
            self.norm_ff_macaron = LayerNorm(size)
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = torch.nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        :param torch.Tensor x_input: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # whether to use macaron style
        if self.macaron_style:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask
        else:
            return x, mask


def _pre_hook_conf(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class ConformerEncoder(torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param bool macaron_style: whether to use macaron style for positionwise layer
    :param bool use_cnn_module: whether to use convolution module
    :param bool zero_triu: whether to zero the upper triangular part of attention matrix
    :param int cnn_module_kernel: kernerl size of convolution module
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        attention_dim=768,
        attention_heads=12,
        linear_units=3072,
        num_blocks=12,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        normalize_before=True,
        concat_after=False,
        macaron_style=True,
        use_cnn_module=True,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        relu_type="swish",
        layer_drop_rate=0.0,
    ):
        """Construct an Encoder object."""
        super(ConformerEncoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook_conf)

        self.embed = torch.nn.Sequential(
            RelPositionalEncoding(attention_dim, positional_dropout_rate)
        )
        self.normalize_before = normalize_before

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, linear_units, dropout_rate)

        encoder_attn_layer = RelPositionMultiHeadedAttention
        encoder_attn_layer_args = (
            attention_heads,
            attention_dim,
            attention_dropout_rate,
            zero_triu,
        )

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                macaron_style,
                positionwise_layer_args
            ),
            layer_drop_rate=0.0,
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        xs = self.embed(xs)

        xs, masks = self.encoders(xs, masks)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        xs, masks = self.embed(xs, masks)

        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache


"""ScorerInterface implementation for CTC."""


class CTCPrefixScorer(BatchPartialScorerInterface):
    """Decoder interface wrapper for CTCPrefixScore."""

    def __init__(self, ctc: torch.nn.Module, eos: int):
        """Initialize class.

        Args:
            ctc (torch.nn.Module): The CTC implementation.
                For example, :class:`espnet.nets.pytorch_backend.ctc.CTC`
            eos (int): The end-of-sequence id.

        """
        self.ctc = ctc
        self.eos = eos
        self.impl = None

    def init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0)).detach().squeeze(0).cpu().numpy()
        # TODO(karita): use CTCPrefixScoreTH
        self.impl = CTCPrefixScore(logp, 0, self.eos, np)
        return 0, self.impl.initial_state()

    def select_state(self, state, i, new_id=None):
        """Select state with relative ids in the main beam search.

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label id to select a state if necessary

        Returns:
            state: pruned state

        """
        if type(state) == tuple:
            if len(state) == 2:  # for CTCPrefixScore
                sc, st = state
                return sc[i], st[i]
            else:  # for CTCPrefixScoreTH (need new_id > 0)
                r, log_psi, f_min, f_max, scoring_idmap = state
                s = log_psi[i, new_id].expand(log_psi.size(1))
                if scoring_idmap is not None:
                    return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
                else:
                    return r[:, :, i, new_id], s, f_min, f_max
        return None if state is None else state[i]

    def score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        prev_score, state = state
        presub_score, new_st = self.impl(y.cpu(), ids.cpu(), state)
        tscore = torch.as_tensor(
            presub_score - prev_score, device=x.device, dtype=x.dtype
        )
        return tscore, (presub_score, new_st)

    def batch_init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0))  # assuming batch_size = 1
        xlen = torch.tensor([logp.size(1)])
        self.impl = CTCPrefixScoreTH(logp, xlen, 0, self.eos)
        return None

    def batch_score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D prefix token
            ids (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        batch_state = (
            (
                torch.stack([s[0] for s in state], dim=2),
                torch.stack([s[1] for s in state]),
                state[0][2],
                state[0][3],
            )
            if state[0] is not None
            else None
        )
        return self.impl(y, batch_state, ids)

    def extend_prob(self, x: torch.Tensor):
        """Extend probs for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            x (torch.Tensor): The encoded feature tensor

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0))
        self.impl.extend_prob(logp)

    def extend_state(self, state):
        """Extend state for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            state: The states of hyps

        Returns: exteded state

        """
        new_state = []
        for s in state:
            new_state.append(self.impl.extend_state(s))

        return new_state


#!/usr/bin/env python3

# Copyright 2018 Mitsubishi Electric Research Labs (Takaaki Hori)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


class CTCPrefixScoreTH(object):
    """Batch processing of CTCPrefixScore

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the label probablities for multiple
    hypotheses simultaneously
    See also Seki et al. "Vectorized Beam Search for CTC-Attention-Based
    Speech Recognition," In INTERSPEECH (pp. 3825-3829), 2019.
    """

    def __init__(self, x, xlens, blank, eos, margin=0):
        """Construct CTC prefix scorer

        :param torch.Tensor x: input label posterior sequences (B, T, O)
        :param torch.Tensor xlens: input lengths (B,)
        :param int blank: blank label id
        :param int eos: end-of-sequence id
        :param int margin: margin parameter for windowing (0 means no windowing)
        """
        # In the comment lines,
        # we assume T: input_length, B: batch size, W: beam width, O: output dim.
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.batch = x.size(0)
        self.input_length = x.size(1)
        self.odim = x.size(2)
        self.dtype = x.dtype
        self.device = (
            torch.device("cuda:%d" % x.get_device())
            if x.is_cuda
            else torch.device("cpu")
        )
        # Pad the rest of posteriors in the batch
        # TODO(takaaki-hori): need a better way without for-loops
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0
        # Reshape input x
        xn = x.transpose(0, 1)  # (B, T, O) -> (T, B, O)
        xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
        self.x = torch.stack([xn, xb])  # (2, T, B, O)
        self.end_frames = torch.as_tensor(xlens) - 1

        # Setup CTC windowing
        self.margin = margin
        if margin > 0:
            self.frame_ids = torch.arange(
                self.input_length, dtype=self.dtype, device=self.device
            )
        # Base indices for index conversion
        self.idx_bh = None
        self.idx_b = torch.arange(self.batch, device=self.device)
        self.idx_bo = (self.idx_b * self.odim).unsqueeze(1)

    def __call__(self, y, state, scoring_ids=None, att_w=None):
        """Compute CTC prefix scores for next labels

        :param list y: prefix label sequences
        :param tuple state: previous CTC state
        :param torch.Tensor pre_scores: scores for pre-selection of hypotheses (BW, O)
        :param torch.Tensor att_w: attention weights to decide CTC window
        :return new_state, ctc_local_scores (BW, O)
        """
        output_length = len(y[0]) - 1  # ignore sos
        last_ids = [yi[-1] for yi in y]  # last output label ids
        n_bh = len(last_ids)  # batch * hyps
        n_hyps = n_bh // self.batch  # assuming each utterance has the same # of hyps
        self.scoring_num = scoring_ids.size(-1) if scoring_ids is not None else 0
        # prepare state info
        if state is None:
            r_prev = torch.full(
                (self.input_length, 2, self.batch, n_hyps),
                self.logzero,
                dtype=self.dtype,
                device=self.device,
            )
            r_prev[:, 1] = torch.cumsum(self.x[0, :, :, self.blank], 0).unsqueeze(2)
            r_prev = r_prev.view(-1, 2, n_bh)
            s_prev = 0.0
            f_min_prev = 0
            f_max_prev = 1
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

        # select input dimensions for scoring
        if self.scoring_num > 0:
            scoring_idmap = torch.full(
                (n_bh, self.odim), -1, dtype=torch.long, device=self.device
            )
            snum = self.scoring_num
            if self.idx_bh is None or n_bh > len(self.idx_bh):
                self.idx_bh = torch.arange(n_bh, device=self.device).view(-1, 1)
            scoring_idmap[self.idx_bh[:n_bh], scoring_ids] = torch.arange(
                snum, device=self.device
            )
            scoring_idx = (
                scoring_ids + self.idx_bo.repeat(1, n_hyps).view(-1, 1)
            ).view(-1)
            x_ = torch.index_select(
                self.x.view(2, -1, self.batch * self.odim), 2, scoring_idx
            ).view(2, -1, n_bh, snum)
        else:
            scoring_ids = None
            scoring_idmap = None
            snum = self.odim
            x_ = self.x.unsqueeze(3).repeat(1, 1, 1, n_hyps, 1).view(2, -1, n_bh, snum)

        # new CTC forward probs are prepared as a (T x 2 x BW x S) tensor
        # that corresponds to r_t^n(h) and r_t^b(h) in a batch.
        r = torch.full(
            (self.input_length, 2, n_bh, snum),
            self.logzero,
            dtype=self.dtype,
            device=self.device,
        )
        if output_length == 0:
            r[0, 0] = x_[0, 0]

        r_sum = torch.logsumexp(r_prev, 1)
        log_phi = r_sum.unsqueeze(2).repeat(1, 1, snum)
        if scoring_ids is not None:
            for idx in range(n_bh):
                pos = scoring_idmap[idx, last_ids[idx]]
                if pos >= 0:
                    log_phi[:, idx, pos] = r_prev[:, 1, idx]
        else:
            for idx in range(n_bh):
                log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

        # decide start and end frames based on attention weights
        if att_w is not None and self.margin > 0:
            f_arg = torch.matmul(att_w, self.frame_ids)
            f_min = max(int(f_arg.min().cpu()), f_min_prev)
            f_max = max(int(f_arg.max().cpu()), f_max_prev)
            start = min(f_max_prev, max(f_min - self.margin, output_length, 1))
            end = min(f_max + self.margin, self.input_length)
        else:
            f_min = f_max = 0
            start = max(output_length, 1)
            end = self.input_length

        # compute forward probabilities log(r_t^n(h)) and log(r_t^b(h))
        for t in range(start, end):
            rp = r[t - 1]
            rr = torch.stack([rp[0], log_phi[t - 1], rp[0], rp[1]]).view(
                2, 2, n_bh, snum
            )
            r[t] = torch.logsumexp(rr, 1) + x_[:, t]

        # compute log prefix probabilities log(psi)
        log_phi_x = torch.cat((log_phi[0].unsqueeze(0), log_phi[:-1]), dim=0) + x_[0]
        if scoring_ids is not None:
            log_psi = torch.full(
                (n_bh, self.odim), self.logzero, dtype=self.dtype, device=self.device
            )
            log_psi_ = torch.logsumexp(
                torch.cat((log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)), dim=0),
                dim=0,
            )
            for si in range(n_bh):
                log_psi[si, scoring_ids[si]] = log_psi_[si]
        else:
            log_psi = torch.logsumexp(
                torch.cat((log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)), dim=0),
                dim=0,
            )

        for si in range(n_bh):
            log_psi[si, self.eos] = r_sum[self.end_frames[si // n_hyps], si]

        # exclude blank probs
        log_psi[:, self.blank] = self.logzero

        return (log_psi - s_prev), (r, log_psi, f_min, f_max, scoring_idmap)

    def index_select_state(self, state, best_ids):
        """Select CTC states according to best ids

        :param state    : CTC state
        :param best_ids : index numbers selected by beam pruning (B, W)
        :return selected_state
        """
        r, s, f_min, f_max, scoring_idmap = state
        # convert ids to BHO space
        n_bh = len(s)
        n_hyps = n_bh // self.batch
        vidx = (best_ids + (self.idx_b * (n_hyps * self.odim)).view(-1, 1)).view(-1)
        # select hypothesis scores
        s_new = torch.index_select(s.view(-1), 0, vidx)
        s_new = s_new.view(-1, 1).repeat(1, self.odim).view(n_bh, self.odim)
        # convert ids to BHS space (S: scoring_num)
        if scoring_idmap is not None:
            snum = self.scoring_num
            hyp_idx = (best_ids // self.odim + (self.idx_b * n_hyps).view(-1, 1)).view(
                -1
            )
            label_ids = torch.fmod(best_ids, self.odim).view(-1)
            score_idx = scoring_idmap[hyp_idx, label_ids]
            score_idx[score_idx == -1] = 0
            vidx = score_idx + hyp_idx * snum
        else:
            snum = self.odim
        # select forward probabilities
        r_new = torch.index_select(r.view(-1, 2, n_bh * snum), 2, vidx).view(
            -1, 2, n_bh
        )
        return r_new, s_new, f_min, f_max

    def extend_prob(self, x):
        """Extend CTC prob.

        :param torch.Tensor x: input label posterior sequences (B, T, O)
        """

        if self.x.shape[1] < x.shape[1]:  # self.x (2,T,B,O); x (B,T,O)
            # Pad the rest of posteriors in the batch
            # TODO(takaaki-hori): need a better way without for-loops
            xlens = [x.size(1)]
            for i, l in enumerate(xlens):
                if l < self.input_length:
                    x[i, l:, :] = self.logzero
                    x[i, l:, self.blank] = 0
            tmp_x = self.x
            xn = x.transpose(0, 1)  # (B, T, O) -> (T, B, O)
            xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
            self.x = torch.stack([xn, xb])  # (2, T, B, O)
            self.x[:, : tmp_x.shape[1], :, :] = tmp_x
            self.input_length = x.size(1)
            self.end_frames = torch.as_tensor(xlens) - 1

    def extend_state(self, state):
        """Compute CTC prefix state.


        :param state    : CTC state
        :return ctc_state
        """

        if state is None:
            # nothing to do
            return state
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

            r_prev_new = torch.full(
                (self.input_length, 2),
                self.logzero,
                dtype=self.dtype,
                device=self.device,
            )
            start = max(r_prev.shape[0], 1)
            r_prev_new[0:start] = r_prev
            for t in range(start, self.input_length):
                r_prev_new[t, 1] = r_prev_new[t - 1, 1] + self.x[0, t, :, self.blank]

            return (r_prev_new, s_prev, f_min_prev, f_max_prev)


class CTCPrefixScore(object):
    """Compute CTC label sequence scores

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously
    """

    def __init__(self, x, blank, eos, xp):
        self.xp = xp
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.input_length = len(x)
        self.x = x

    def initial_state(self):
        """Obtain an initial CTC state

        :return: CTC state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        r = self.xp.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]
        for i in range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def __call__(self, y, cs, r_prev):
        """Compute CTC prefix scores for next labels

        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        output_length = len(y) - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = self.xp.ndarray((self.input_length, 2, len(cs)), dtype=np.float32)
        xs = self.x[:, cs]
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            r[output_length - 1] = self.logzero

        # prepare forward probabilities for the last label
        r_sum = self.xp.logaddexp(
            r_prev[:, 0], r_prev[:, 1]
        )  # log(r_t^n(g) + r_t^b(g))
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = self.xp.ndarray((self.input_length, len(cs)), dtype=np.float32)
            for i in range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilities log(psi)
        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        for t in range(start, self.input_length):
            r[t, 0] = self.xp.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = (
                self.xp.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.x[t, self.blank]
            )
            log_psi = self.xp.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        # get P(...eos|X) that ends with the prefix itself
        eos_pos = self.xp.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

        # exclude blank probs
        blank_pos = self.xp.where(cs == self.blank)[0]
        if len(blank_pos) > 0:
            log_psi[blank_pos] = self.logzero

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, self.xp.rollaxis(r, 2)


class CTC(torch.nn.Module):
    """CTC module

    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    :param bool reduce: reduce the CTC loss into a scalar
    """

    def __init__(self, odim, eprojs, dropout_rate, reduce=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.loss = None
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.probs = None  # for visualization

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type, zero_infinity=True)
        self.ignore_id = -1
        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen):
        th_pred = th_pred.log_softmax(2)
        with torch.backends.cudnn.flags(deterministic=True):
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
        # Batch-size average
        loss = loss / th_pred.size(1)
        return loss

    def forward(self, hs_pad, hlens, ys_pad):
        """CTC forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        # zero padding for hs
        ys_hat = self.ctc_lo(self.dropout(hs_pad))
        ys_hat = ys_hat.transpose(0, 1)

        olens = to_device(ys_hat, torch.LongTensor([len(s) for s in ys]))
        hlens = hlens.long()
        ys_pad = torch.cat(ys)  # without this the code breaks for asr_mix
        self.loss = self.loss_fn(ys_hat, ys_pad, hlens, olens)

        if self.reduce:
            self.loss = self.loss.sum()

        return self.loss, ys_hat

    def softmax(self, hs_pad):
        """softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        self.probs = F.softmax(self.ctc_lo(hs_pad), dim=-1)
        return self.probs

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=-1)

    def argmax(self, hs_pad):
        """argmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: argmax applied 2d tensor (B, Tmax)
        :rtype: torch.Tensor
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=-1)

    def forced_align(self, h, y, blank_id=0):
        """forced alignment.

        :param torch.Tensor h: hidden state sequence, 2d tensor (T, D)
        :param torch.Tensor y: id sequence tensor 1d tensor (L)
        :param int y: blank symbol index
        :return: best alignment results
        :rtype: list
        """

        def interpolate_blank(label, blank_id=0):
            """Insert blank token between every two label token."""
            label = np.expand_dims(label, 1)
            blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
            label = np.concatenate([blanks, label], axis=1)
            label = label.reshape(-1)
            label = np.append(label, label[0])
            return label

        lpz = self.log_softmax(h)
        lpz = lpz.squeeze(0)

        y_int = interpolate_blank(y, blank_id)

        logdelta = np.zeros((lpz.size(0), len(y_int))) - 100000000000.0  # log of zero
        state_path = (
            np.zeros((lpz.size(0), len(y_int)), dtype=np.int16) - 1
        )  # state path

        logdelta[0, 0] = lpz[0][y_int[0]]
        logdelta[0, 1] = lpz[0][y_int[1]]

        for t in range(1, lpz.size(0)):
            for s in range(len(y_int)):
                if y_int[s] == blank_id or s < 2 or y_int[s] == y_int[s - 2]:
                    candidates = np.array([logdelta[t - 1, s], logdelta[t - 1, s - 1]])
                    prev_state = [s, s - 1]
                else:
                    candidates = np.array(
                        [
                            logdelta[t - 1, s],
                            logdelta[t - 1, s - 1],
                            logdelta[t - 1, s - 2],
                        ]
                    )
                    prev_state = [s, s - 1, s - 2]
                logdelta[t, s] = np.max(candidates) + lpz[t][y_int[s]]
                state_path[t, s] = prev_state[np.argmax(candidates)]

        state_seq = -1 * np.ones((lpz.size(0), 1), dtype=np.int16)

        candidates = np.array(
            [logdelta[-1, len(y_int) - 1], logdelta[-1, len(y_int) - 2]]
        )
        prev_state = [len(y_int) - 1, len(y_int) - 2]
        state_seq[-1] = prev_state[np.argmax(candidates)]
        for t in range(lpz.size(0) - 2, -1, -1):
            state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

        output_state_seq = []
        for t in range(0, lpz.size(0)):
            output_state_seq.append(y_int[state_seq[t, 0]])

        return output_state_seq

    def forced_align_batch(self, hs_pad, ys_pad, ilens, blank_id=0):
        """forced alignment with batch processing.

        :param torch.Tensor hs_pad: hidden state sequence, 3d tensor (T, B, D)
        :param torch.Tensor ys_pad: id sequence tensor 2d tensor (B, L)
        :param torch.Tensor ilens: Input length of each utterance (B,)
        :param int blank_id: blank symbol index
        :return: best alignment results
        :rtype: list of numpy.array
        """

        def interpolate_blank(label, olens_int):
            """Insert blank token between every two label token."""
            lab_len = label.shape[1] * 2 + 1
            label_out = np.full((label.shape[0], lab_len), blank_id, dtype=np.int64)
            label_out[:, 1::2] = label
            for b in range(label.shape[0]):
                label_out[b, olens_int[b] * 2 + 1 :] = self.ignore_id
            return label_out

        neginf = float("-inf")  # log of zero
        # lpz = self.log_softmax(hs_pad).cpu().detach().numpy()
        # hs_pad = hs_pad.transpose(1,0)
        lpz = F.log_softmax(hs_pad, dim=-1).cpu().detach().numpy()
        ilens = ilens.cpu().detach().numpy()

        ys_pad = ys_pad.cpu().detach().numpy()
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        olens = np.array([len(s) for s in ys])
        olens_int = olens * 2 + 1
        ys_int = interpolate_blank(ys_pad, olens_int)

        Tmax, B, _ = lpz.shape
        Lmax = ys_int.shape[-1]
        logdelta = np.full((Tmax, B, Lmax), neginf, dtype=lpz.dtype)
        state_path = -np.ones(logdelta.shape, dtype=np.int16)  # state path

        b_indx = np.arange(B, dtype=np.int64)
        t_0 = np.zeros(B, dtype=np.int64)
        logdelta[0, :, 0] = lpz[t_0, b_indx, ys_int[:, 0]]
        logdelta[0, :, 1] = lpz[t_0, b_indx, ys_int[:, 1]]

        s_indx_mat = np.arange(Lmax)[None, :].repeat(B, 0)
        notignore_mat = ys_int != self.ignore_id
        same_lab_mat = np.zeros((B, Lmax), dtype=bool)
        same_lab_mat[:, 3::2] = ys_int[:, 3::2] == ys_int[:, 1:-2:2]
        Lmin = olens_int.min()
        for t in range(1, Tmax):
            s_start = max(0, Lmin - (Tmax - t) * 2)
            s_end = min(Lmax, t * 2 + 2)
            candidates = np.full((B, Lmax, 3), neginf, dtype=logdelta.dtype)
            candidates[:, :, 0] = logdelta[t - 1, :, :]
            candidates[:, 1:, 1] = logdelta[t - 1, :, :-1]
            candidates[:, 3::2, 2] = logdelta[t - 1, :, 1:-2:2]
            candidates[same_lab_mat, 2] = neginf
            candidates_ = candidates[:, s_start:s_end, :]
            idx = candidates_.argmax(-1)
            b_i, s_i = np.ogrid[:B, : idx.shape[-1]]
            nignore = notignore_mat[:, s_start:s_end]
            logdelta[t, :, s_start:s_end][nignore] = (
                candidates_[b_i, s_i, idx][nignore]
                + lpz[t, b_i, ys_int[:, s_start:s_end]][nignore]
            )
            s = s_indx_mat[:, s_start:s_end]
            state_path[t, :, s_start:s_end][nignore] = (s - idx)[nignore]

        alignments = []
        prev_states = logdelta[
            ilens[:, None] - 1,
            b_indx[:, None],
            np.stack([olens_int - 2, olens_int - 1], -1),
        ].argmax(-1)
        for b in range(B):
            T, L = ilens[b], olens_int[b]
            prev_state = prev_states[b] + L - 2
            ali = np.empty(T, dtype=ys_int.dtype)
            ali[T - 1] = ys_int[b, prev_state]
            for t in range(T - 2, -1, -1):
                prev_state = state_path[t + 1, b, prev_state]
                ali[t] = ys_int[b, prev_state]
            alignments.append(ali)

        return alignments


#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


class AutoAVSR(torch.nn.Module):
    def __init__(self, odim=5049, modality="video", ctc_weight=0.1, ignore_id=-1):
        super().__init__()

        self.modality = modality
        if modality == "audio":
            self.frontend = audio_resnet()
        elif modality == "video":
            self.frontend = video_resnet()

        self.proj_encoder = torch.nn.Linear(512, 768)

        self.encoder = ConformerEncoder(
            attention_dim=768,
            attention_heads=12,
            linear_units=3072,
            num_blocks=12,
            cnn_module_kernel=31,
        )

        # self.decoder = TransformerDecoder(
        #     odim=odim,
        #     attention_dim=768,
        #     attention_heads=12,
        #     linear_units=3072,
        #     num_blocks=6,
        # )
        #
        # self.blank = 0
        # self.sos = odim - 1
        # self.eos = odim - 1
        # self.odim = odim
        # self.ignore_id = ignore_id
        #
        # # loss
        # self.ctc_weight = ctc_weight
        # self.ctc = CTC(odim, 768, 0.1, reduce=True)
        # self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, 0.1, False)

    # def scorers(self):
    #     return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    # def forward(self, x, lengths, label):
    #     if self.modality == "audio":
    #         lengths = torch.div(lengths, 640, rounding_mode="trunc")
    #
    #     padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
    #
    #     x = self.frontend(x)
    #     x = self.proj_encoder(x)
    #     x, _ = self.encoder(x, padding_mask)
    #
    #     # ctc loss
    #     loss_ctc, ys_hat = self.ctc(x, lengths, label)
    #
    #     # decoder loss
    #     ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
    #     ys_mask = target_mask(ys_in_pad, self.ignore_id)
    #     pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, padding_mask)
    #     loss_att = self.criterion(pred_pad, ys_out_pad)
    #     loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
    #
    #     acc = th_accuracy(
    #         pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
    #     )
    #
    #     return loss, loss_ctc, loss_att, acc

    def _encode_raw(self, sample, mask=None, lengths=None):
        """Encode one batch. Pad frames still leak through the unmasked convs.

        Prefer ``encode``, which zeroes those frames before each time-mixing conv.
        """
        if len(sample.shape) != 5:
            raise ValueError(f"The input shape should be [b, t, 1, 88, 88], it is {sample.shape}")
        x = self.frontend(sample)
        x = self.proj_encoder(x)
        if lengths is not None:
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths, device=x.device, dtype=torch.long)
            mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
        enc_feat, _ = self.encoder(x, mask)
        return enc_feat, x

    def _pad_zero_pre_hook(self, module, args):
        """Zero pad positions on a time-mixing conv while ``_pad_keep`` is set."""
        keep = self._pad_keep
        if keep is None:
            return None
        x = args[0]
        if x.dim() == 5:  # frontend Conv3d: [b, C, t, H, W]
            k = keep.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3:  # conformer depthwise Conv1d: [b, C, t]
            k = keep
        else:
            raise RuntimeError(
                f"pad-zero hook got unexpected input rank {x.dim()} (shape {tuple(x.shape)})"
            )
        k = k.to(device=x.device, dtype=x.dtype)
        return (x * k,) + tuple(args[1:])

    def _install_pad_zero_hooks(self):
        """Hook the frontend 3D conv and each conformer depthwise conv.

        Those are the time-mixing convs. Attention is already masked, and the
        ResNet trunk is per-frame. Hooks stay installed and no-op until
        ``_pad_keep`` is set.
        """
        if getattr(self, "_pad_zero_handles", None):
            return
        targets = [m for m in self.frontend.modules() if isinstance(m, torch.nn.Conv3d)]
        targets += [
            layer.conv_module.depthwise_conv
            for layer in self.encoder.encoders
            if getattr(layer, "conv_module", None) is not None
        ]
        self._pad_zero_handles = [
            module.register_forward_pre_hook(self._pad_zero_pre_hook) for module in targets
        ]
        self._pad_keep = None

    def encode(self, sample, mask=None, lengths=None):
        """Batched encode whose valid frames match a single-sample encode.

        The attention mask does not cover the conformer depthwise convs
        (kernel 31) or the frontend Conv3d, so a padded tail leaks into valid
        frames. This zeroes that tail at each of those convs.

        In eval mode the valid frames match encoding each sample alone, up to
        normal fp32 noise. In train mode they do not: BatchNorm statistics
        still include the zeroed pad frames.

        Args:
            sample (torch.Tensor): [b, t, 1, 88, 88]
            mask: Ignored. The mask is built from ``lengths``.
            lengths (list[int] | torch.Tensor | None): Frames before padding.
                None means every sample is full length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``enc_feat`` [b, t, 768], with
            zeros past each sample's length, and the projected sequence (pad
            frames included).
        """
        if len(sample.shape) != 5:
            raise ValueError(f"The input shape should be [b, t, 1, 88, 88], it is {sample.shape}")
        b, t = sample.shape[0], sample.shape[1]
        if lengths is None:
            lengths = [t] * b
        elif isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()
        lengths = [int(length) for length in lengths]
        if len(lengths) != b or any(length <= 0 or length > t for length in lengths):
            raise ValueError(f"lengths must be b ints in (0, {t}], got {lengths}")

        keep = torch.zeros(b, 1, t, device=sample.device)
        for i, length in enumerate(lengths):
            keep[i, 0, :length] = 1.0

        self._install_pad_zero_hooks()
        self._pad_keep = keep
        try:
            enc_feat, projected = self._encode_raw(sample, lengths=lengths)
        finally:
            self._pad_keep = None
        enc_feat = enc_feat * keep[:, 0].unsqueeze(-1).to(enc_feat)
        return enc_feat, projected


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positional Encoding Module."""


def _pre_hook_emb(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.
    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.
    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook_emb)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.
    See Sec. 3.2  https://arxiv.org/abs/1809.08895
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(x)


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        # self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Label smoothing module."""


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
    ):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Layer normalization module."""


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Mask module."""


def subsequent_mask(size, device="cpu", dtype=torch.bool):
    """Create mask for subsequent steps (size, size).

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)


def target_mask(ys_in_pad, ignore_id):
    """Create mask for decoder self-attention.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int ignore_id: index of padding
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    """
    ys_mask = ys_in_pad != ignore_id
    m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
    return ys_mask.unsqueeze(-2) & m


# -*- coding: utf-8 -*-

"""Network related utility tools."""


def to_device(m, x):
    """Send tensor into the device of the module.

    Args:
        m (torch.nn.Module): Torch module.
        x (Tensor): Torch tensor.

    Returns:
        Tensor: Torch tensor located in the same place as torch module.

    """
    if isinstance(m, torch.nn.Module):
        device = next(m.parameters()).device
    elif isinstance(m, torch.Tensor):
        device = m.device
    else:
        raise TypeError(
            "Expected torch.nn.Module or torch.tensor, " f"bot got: {type(m)}"
        )
    return x.to(device)


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)


def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax, D).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)
    ).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


def rename_state_dict(
    old_prefix: str, new_prefix: str, state_dict: Dict[str, torch.Tensor]
):
    """Replace keys of old prefix with new prefix in state dict."""
    # need this list not to break the dict iterator
    old_keys = [k for k in state_dict if k.startswith(old_prefix)]
    if len(old_keys) > 0:
        logging.warning(f"Rename: {old_prefix} -> {new_prefix}")
    for k in old_keys:
        v = state_dict.pop(k)
        new_k = k.replace(old_prefix, new_prefix)
        state_dict[new_k] = v


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, hidden_units, dropout_rate):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward funciton."""
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Repeat the same layer definition."""


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def __init__(self, *args, layer_drop_rate=0.0):
        """Initialize MultiSequential with layer_drop.

        Args:
            layer_drop_rate (float): Probability of dropping out each fn (layer).

        """
        super(MultiSequential, self).__init__(*args)
        self.layer_drop_rate = layer_drop_rate

    def forward(self, *args):
        """Repeat."""
        _probs = torch.empty(len(self)).uniform_()
        for idx, m in enumerate(self):
            if not self.training or (_probs[idx] >= self.layer_drop_rate):
                args = m(*args)
        return args


def repeat(N, fn, layer_drop_rate=0.0):
    """Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.
        layer_drop_rate (float): Probability of dropping out each fn (layer).

    Returns:
        MultiSequential: Repeated model instance.

    """
    return MultiSequential(*[fn(n) for n in range(N)], layer_drop_rate=layer_drop_rate)


def conv3x3_1d(in_planes, out_planes, stride=1):
    """conv3x3.
    :param in_planes: int, number of channels in the input sequence.
    :param out_planes: int,  number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block_1d(inplanes, outplanes, stride):
    """downsample_basic_block.
    :param inplanes: int, number of channels in the input sequence.
    :param outplanes: int, number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return nn.Sequential(
        nn.Conv1d(
            inplanes,
            outplanes,
            kernel_size=1,
            stride=stride,
            bias=False,
        ),
        nn.BatchNorm1d(outplanes),
    )


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        relu_type="relu",
    ):
        """__init__.
        :param inplanes: int, number of channels in the input sequence.
        :param planes: int,  number of channels produced by the convolution.
        :param stride: int, size of the convolving kernel.
        :param downsample: boolean, if True, the temporal resolution is downsampled.
        :param relu_type: str, type of activation function.
        """
        super(BasicBlock1D, self).__init__()

        assert relu_type in ["relu", "prelu", "swish"]

        self.conv1 = conv3x3_1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)

        # type of ReLU is an input option
        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        elif relu_type == "swish":
            self.relu1 = nn.SiLU(inplace=True)
            self.relu2 = nn.SiLU(inplace=True)
        else:
            raise NotImplementedError
        # --------

        self.conv2 = conv3x3_1d(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T)
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        block,
        layers,
        relu_type="swish",
        a_upsample_ratio=1,
    ):
        """__init__.
        :param block: torch.nn.Module, class of blocks.
        :param layers: List, customised layers in each block.
        :param relu_type: str, type of activation function.
        :param a_upsample_ratio: int, The ratio related to the \
            temporal resolution of output features of the frontend. \
            a_upsample_ratio=1 produce features with a fps of 25.
        """
        super(ResNet1D, self).__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.downsample_block = downsample_basic_block_1d
        self.a_upsample_ratio = a_upsample_ratio

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.inplanes,
            kernel_size=80,
            stride=4,
            padding=38,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(self.inplanes)

        if relu_type == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu = nn.PReLU(num_parameters=self.inplanes)
        elif relu_type == "swish":
            self.relu = nn.SiLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(
            kernel_size=20 // self.a_upsample_ratio,
            stride=20 // self.a_upsample_ratio,
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer.
        :param block: torch.nn.Module, class of blocks.
        :param planes: int,  number of channels produced by the convolution.
        :param blocks: int, number of layers in a block.
        :param stride: int, size of the convolving kernel.
        """

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                relu_type=self.relu_type,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    relu_type=self.relu_type,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x


class Conv1dResNet(nn.Module):
    """Conv1dResNet"""

    def __init__(self, relu_type="swish", a_upsample_ratio=1):
        """__init__.
        :param relu_type: str, Activation function used in an audio front-end.
        :param a_upsample_ratio: int, The ratio related to the \
            temporal resolution of output features of the frontend. \
            a_upsample_ratio=1 produce features with a fps of 25.
        """

        super(Conv1dResNet, self).__init__()
        self.a_upsample_ratio = a_upsample_ratio
        self.trunk = ResNet1D(
            BasicBlock1D,
            [2, 2, 2, 2],
            relu_type=relu_type,
            a_upsample_ratio=a_upsample_ratio,
        )

    def forward(self, xs_pad):
        """forward.
        :param xs_pad: torch.Tensor, batch of padded input sequences (B, Tmax, idim)
        """
        B, T, C = xs_pad.size()
        xs_pad = xs_pad[:, : T // 640 * 640, :]
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = self.trunk(xs_pad)
        # -- from B x C x T to B x T x C
        xs_pad = xs_pad.transpose(1, 2)
        return xs_pad


def audio_resnet():
    return Conv1dResNet()


def conv3x3(in_planes, out_planes, stride=1):
    """conv3x3.
    :param in_planes: int, number of channels in the input sequence.
    :param out_planes: int,  number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block(inplanes, outplanes, stride):
    """downsample_basic_block.
    :param inplanes: int, number of channels in the input sequence.
    :param outplanes: int, number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return nn.Sequential(
        nn.Conv2d(
            inplanes,
            outplanes,
            kernel_size=1,
            stride=stride,
            bias=False,
        ),
        nn.BatchNorm2d(outplanes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        relu_type="swish",
    ):
        """__init__.
        :param inplanes: int, number of channels in the input sequence.
        :param planes: int,  number of channels produced by the convolution.
        :param stride: int, size of the convolving kernel.
        :param downsample: boolean, if True, the temporal resolution is downsampled.
        :param relu_type: str, type of activation function.
        """
        super(BasicBlock, self).__init__()

        assert relu_type in ["relu", "prelu", "swish"]

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        elif relu_type == "swish":
            self.relu1 = nn.SiLU(inplace=True)
            self.relu2 = nn.SiLU(inplace=True)
        else:
            raise NotImplementedError
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        relu_type="swish",
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.downsample_block = downsample_basic_block

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer.
        :param block: torch.nn.Module, class of blocks.
        :param planes: int,  number of channels produced by the convolution.
        :param blocks: int, number of layers in a block.
        :param stride: int, size of the convolving kernel.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                relu_type=self.relu_type,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    relu_type=self.relu_type,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class Conv3dResNet(nn.Module):
    """Conv3dResNet module"""

    def __init__(self, backbone_type="resnet", relu_type="swish"):
        """__init__.
        :param backbone_type: str, the type of a visual front-end.
        :param relu_type: str, activation function used in an audio front-end.
        """
        super(Conv3dResNet, self).__init__()

        self.backbone_type = backbone_type

        self.frontend_nout = 64
        self.trunk = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            relu_type=relu_type,
        )

        # -- frontend3D
        if relu_type == "relu":
            frontend_relu = nn.ReLU(True)
        elif relu_type == "prelu":
            frontend_relu = nn.PReLU(self.frontend_nout)
        elif relu_type == "swish":
            frontend_relu = nn.SiLU(inplace=True)

        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )

    def forward(self, xs_pad):
        """forward.
        :param xs_pad: torch.Tensor, batch of padded input sequences.
        """
        # -- include Channel dimension
        xs_pad = xs_pad.transpose(2, 1)
        B, C, T, H, W = xs_pad.size()
        xs_pad = self.frontend3D(xs_pad)
        Tnew = xs_pad.shape[2]  # outpu should be B x C2 x Tnew x H x W
        xs_pad = threeD_to_2D_tensor(xs_pad)
        xs_pad = self.trunk(xs_pad)
        xs_pad = xs_pad.view(B, Tnew, xs_pad.size(1))
        return xs_pad


def video_resnet():
    return Conv3dResNet()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""


class DecoderLayer(torch.nn.Module):
    """Single decoder layer module.
    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward: feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = torch.nn.Linear(size + size, size)
            self.concat_linear2 = torch.nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Compute decoded features.
        Args:
            tgt (torch.Tensor):
                decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask


def _pre_hook_tr_dec(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "output_norm.", prefix + "after_norm.", state_dict)


class TransformerDecoder(BatchScorerInterface, torch.nn.Module):
    """Transfomer decoder module.

    :param int odim: output dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate for attention
    :param str or torch.nn.Module input_layer: input layer type
    :param bool use_output_layer: whether to use output layer
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
        self,
        odim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.1,
        src_attention_dropout_rate=0.1,
        input_layer="embed",
        use_output_layer=True,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        layer_drop_rate=0.0,
    ):
        """Construct an Decoder object."""
        torch.nn.Module.__init__(self)
        self._register_load_state_dict_pre_hook(_pre_hook_tr_dec)
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer, pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
            layer_drop_rate,
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, odim)
        else:
            self.output_layer = None

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward decoder.
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
                                 if input_layer == "embed"
                                 input tensor (batch, maxlen_out, #mels)
                                 in the other cases
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask,  (batch, maxlen_in)
                                         dtype=torch.uint8 in PyTorch 1.2-
                                         dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :return x: decoded token score before softmax (batch, maxlen_out, token)
                   if use_output_layer is True,
                   final block outputs (batch, maxlen_out, attention_dim)
                   in the other cases
        :rtype: torch.Tensor
        :return tgt_mask: score mask before softmax (batch, maxlen_out)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        """Forward one step.
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param List[torch.Tensor] cache:
            cached output list of (batch, max_time_out-1, size)
        :return y, cache: NN output value and cache per `self.decoders`.
            `y.shape` is (batch, maxlen_out, token)
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).
        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][l] for b in range(n_batch)])
                for l in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[l][b] for l in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
