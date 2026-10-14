#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Policy Decoder Module for ACT
- Action Queries: k=50 learnable embeddings + 1D Sinusoidal Temporal PE
- Transformer Decoder (4 Layers) with Multi-Head Self-Attn & Cross-Attn over Memory Tokens:
    [N_vis Visual Tokens + 1 qpos Token + 1 Latent z Token]
- Cross-Attention Weight Extraction: Hỗ trợ trích xuất ma trận Attention cho Hàm mất mát Phương pháp 2
- Action Head: Linear 512 -> 16 Joint Targets
"""

from typing import Optional, Tuple, List, Union
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding1D(nn.Module):
    """
    1D Sinusoidal Temporal Positional Encoding (Fixed)
    Mã hóa thứ tự thời gian cho chuỗi dự đoán 50 bước tương lai [t, t+1, ..., t+49].
    """
    def __init__(self, d_model: int = 512, max_len: int = 100, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(temperature) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, length: int) -> torch.Tensor:
        """
        Trả về PE cho độ dài length: [1, length, d_model]
        """
        return self.pe[:, :length, :]


class ACTDecoderLayer(nn.TransformerDecoderLayer):
    """
    Subclass của nn.TransformerDecoderLayer hỗ trợ trả về ma trận Cross-Attention weights.
    Bảo toàn 100% cấu trúc tham số (state_dict keys: self_attn, multihead_attn, linear1, ...)
    đảm bảo tương thích tuyệt đối với các checkpoint đã train và ONNX export.
    """
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        # 1. Self-Attention Block
        x = tgt
        sa_out = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
        x = self.norm1(x + sa_out)

        # 2. Cross-Attention Block
        if return_attn:
            attn_out, attn_weights = self.multihead_attn(
                x,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            x = self.norm2(x + self.dropout2(attn_out))
            x = self.norm3(x + self._ff_block(x))
            return x, attn_weights
        else:
            mha_out = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            x = self.norm2(x + mha_out)
            x = self.norm3(x + self._ff_block(x))
            return x


class ACTTransformerDecoder(nn.Module):
    """
    Transformer Decoder container gồm danh sách ACTDecoderLayer.
    Bảo toàn 100% quy ước đặt tên 'layers.0...', 'layers.1...', v.v.
    """
    def __init__(self, decoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        return_attn: bool = False,
    ):
        output = tgt
        attn_weights_list = []
        for mod in self.layers:
            if return_attn:
                output, attn_w = mod(output, memory, return_attn=True)
                attn_weights_list.append(attn_w)
            else:
                output = mod(output, memory, return_attn=False)

        if return_attn:
            return output, attn_weights_list
        return output


class TransformerPolicyDecoder(nn.Module):
    """
    Transformer Policy Decoder cho ACT:
    - 50 Action Queries tự học biểu diễn ý định hành động trong tương lai.
    - 1D Temporal Sinusoidal PE cung cấp nhãn thời gian thứ tự 0..49.
    - 4 Layers Transformer Decoder với Multi-Head Self-Attention và Cross-Attention.
    - Memory tokens gồm N_mem phần tử: [N_vis Visual + 1 qpos + 1 z].
    - Hỗ trợ trích xuất ma trận Cross-Attention A khi return_attn=True phục vụ Hàm mất mát Phương pháp 2.
    - Action Linear Head chiếu từ d_model (512) ra 16 góc mục tiêu cho 16 động cơ Damiao.
    """
    def __init__(
        self,
        d_model: int = 512,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
        chunk_size: int = 50,
        action_dim: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # 50 Action Queries học được (Learnable Parameter)
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        nn.init.normal_(self.action_queries, std=0.02)

        # 1D Temporal Sinusoidal PE (Fixed)
        self.temporal_pe = SinusoidalPositionEmbedding1D(d_model=d_model, max_len=chunk_size + 10)

        # 4 tầng Transformer Decoder (Custom ACT Decoder hỗ trợ trích xuất Cross-Attention)
        decoder_layer = ACTDecoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.decoder = ACTTransformerDecoder(decoder_layer, num_layers=num_layers)

        # Action Linear Head: 512 -> 16 góc mục tiêu
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, memory: torch.Tensor, return_attn: bool = False):
        """
        Inputs:
            memory: [B, N_mem, d_model] - Memory tokens (N_vis Visual + 1 qpos + 1 z)
            return_attn: bool - Nếu True, trả về kèm danh sách ma trận Cross-Attention các tầng
        Outputs:
            pred_actions: [B, chunk_size, action_dim] - Quỹ đạo 50 bước góc cho 16 động cơ
            (Tùy chọn) attn_weights: List[[B, chunk_size, N_mem]] nếu return_attn=True
        """
        B = memory.shape[0]

        # 1. Khởi tạo Action Queries với Temporal PE
        queries = self.action_queries.repeat(B, 1, 1) # [B, chunk_size, d_model]
        pe = self.temporal_pe(self.chunk_size)        # [1, chunk_size, d_model]
        tgt = queries + pe                            # [B, chunk_size, d_model]

        # 2. Giải mã qua các tầng Transformer Decoder (Cross-Attention với Memory)
        if return_attn:
            decoded, attn_weights = self.decoder(tgt=tgt, memory=memory, return_attn=True)
            pred_actions = self.action_head(decoded)  # [B, chunk_size, action_dim]
            return pred_actions, attn_weights
        else:
            decoded = self.decoder(tgt=tgt, memory=memory, return_attn=False)
            pred_actions = self.action_head(decoded)  # [B, chunk_size, action_dim]
            return pred_actions
