#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Attention Loss Module for ACT Training
Phương pháp 2: Loss phạt Cross-Attention (Visual Attention Alignment / Spatial Foreground Loss)

Trong TransformerPolicyDecoder, mỗi tầng Decoder tính Cross-Attention giữa 50 action_queries
và memory tokens ([N_vis Visual Tokens, 1 qpos Token, 1 z Token]).

Bao gồm:
2A. Visual Attention Ratio Loss (Chống lấn át Modality / Proprioceptive Overfitting):
    Gọi A in R^{B x Q x N_mem} là ma trận trọng số Cross-Attention (sau Softmax, trung bình các heads).
    A_vis = sum_{i=0}^{N_vis-1} A_{., ., i} là tổng sự chú ý dành cho thị giác.
    A_state = A_{., ., N_vis} + A_{., ., N_vis+1} là sự chú ý vào qpos + z.
    Phạt Hinge Loss nếu tỷ lệ chú ý vào ảnh thấp hơn ngưỡng tau (mặc định tau = 0.6):
    L_attn_focus = max(0, tau - A_vis)

2B. Spatial Foreground Attention Loss (Tập trung Attention vào vùng bàn thao tác):
    Dùng kênh Depth sẵn có trong ảnh RGB-D hạ độ phân giải về lưới patch (ví dụ 15x20 = 300 patches)
    để tạo mask vùng tiền cảnh M_fg in [0, 1]^{N_vis}.
    Phạt toàn bộ trọng số Cross-Attention rơi vào các patch nền/tường (1 - M_fg):
    L_bg_suppress = sum_{i=0}^{N_vis-1} A_{., ., i} * (1 - M_fg, i)

* LƯU Ý KỸ THUẬT:
Nhánh này CHỈ được kích hoạt trong lúc huấn luyện (training) khi return_aux=True.
Khi Inference hoặc xuất ONNX (50Hz), return_attn=False hoàn toàn,
đảm bảo 100% không ảnh hưởng tới tốc độ suy luận 50Hz của robot.
"""

from typing import List, Tuple, Dict, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLossModule(nn.Module):
    """
    Module tính toán hàm mất mát Cross-Attention Alignment cho ACT:
    - 2A: Visual Attention Focus Loss (chống lãng quên thị giác, ép mạng quan sát ảnh)
    - 2B: Spatial Background Suppression Loss (triệt tiêu sự chú ý vào nền tĩnh, tường, sàn)
    """
    def __init__(
        self,
        tau: float = 0.6,
        focus_weight: float = 1.0,
        bg_weight: float = 1.0,
        binary_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.tau = tau
        self.focus_weight = focus_weight
        self.bg_weight = bg_weight
        self.binary_threshold = binary_threshold

    def compute_foreground_mask(
        self,
        image: torch.Tensor,
        spatial_shape: Tuple[int, int],
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Tạo Foreground Proximity Mask M_fg từ kênh Depth (kênh index 3 của image [B, 4, H, W]).
        Trong chuẩn tiền xử lý:
        - depth_channel in [0, 1]: 0.0 là cự ly gần nhất 0.2m (tay robot, vật thể trên bàn)
        - 1.0 là cự ly xa nhất 1.2m (tường, nền sàn)
        Proximity = clamp(1.0 - depth_channel, 0.0, 1.0).
        Downsample về kích thước lưới patch (H_f, W_f) và làm phẳng thành [B, 1, N_vis].
        """
        depth_channel = image[:, 3:4, :, :] # [B, 1, H, W]
        proximity = torch.clamp(1.0 - depth_channel, 0.0, 1.0) # [B, 1, H, W]

        # Downsample về kích thước đặc trưng không gian
        m_fg = F.interpolate(
            proximity,
            size=spatial_shape,
            mode="bilinear",
            align_corners=False,
        ) # [B, 1, H_f, W_f]

        if threshold is not None:
            m_fg = (m_fg >= threshold).float()

        m_fg_flat = m_fg.flatten(2) # [B, 1, N_vis]
        return m_fg_flat

    def infer_spatial_shape(self, num_visual_tokens: int, image: Optional[torch.Tensor] = None) -> Tuple[int, int]:
        """
        Suy luận kích thước không gian (H_f, W_f) từ số lượng visual tokens hoặc kích thước ảnh.
        - Ảnh 640x480: H_f = 480 / 32 = 15, W_f = 640 / 32 = 20 -> 300 patches
        - Ảnh 240x424: H_f = 7, W_f = 13 -> 91 patches
        """
        if image is not None:
            H, W = image.shape[2], image.shape[3]
            H_f = max(1, H // 32)
            W_f = max(1, W // 32)
            if H_f * W_f == num_visual_tokens:
                return (H_f, W_f)

        if num_visual_tokens == 300:
            return (15, 20)
        elif num_visual_tokens == 91:
            return (7, 13)
        else:
            # Ước lượng tỷ lệ 4:3 hoặc gần vuông
            side = int(math.isqrt(num_visual_tokens))
            for h in range(side, 0, -1):
                if num_visual_tokens % h == 0:
                    return (h, num_visual_tokens // h)
            return (1, num_visual_tokens)

    def forward(
        self,
        attn_weights: Union[List[torch.Tensor], torch.Tensor],
        num_visual_tokens: int,
        image: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        focus_weight: Optional[float] = None,
        bg_weight: Optional[float] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tính toán các thành phần hàm mất mát Cross-Attention.

        Inputs:
            attn_weights: Danh sách các ma trận Attention [B, Q, N_mem] từ mỗi tầng Decoder,
                          hoặc Tensor [num_layers, B, Q, N_mem]
            num_visual_tokens: Số visual tokens N_vis (phần còn lại sau đó là qpos + z)
            image: Tensor [B, 4, H, W] để trích xuất mask chiều sâu (Depth Foreground)
            tau: Ngưỡng Hinge Loss cho tỷ lệ chú ý thị giác (mặc định self.tau = 0.6)
            focus_weight: Trọng số cho 2A Focus Loss
            bg_weight: Trọng số cho 2B Background Suppression Loss
            spatial_shape: (H_f, W_f) tùy chọn

        Outputs:
            dict chứa:
                - "attn_loss": Tổng mất mát Cross-Attention kết hợp
                - "attn_focus_loss": 2A Visual Attention Ratio Loss
                - "attn_bg_loss": 2B Spatial Background Suppression Loss
                - "attn_vis_ratio": Tỷ lệ chú ý trung bình vào Visual Tokens
        """
        if attn_weights is None or len(attn_weights) == 0:
            zero_val = torch.tensor(0.0)
            return {
                "attn_loss": zero_val,
                "attn_focus_loss": zero_val,
                "attn_bg_loss": zero_val,
                "attn_vis_ratio": zero_val,
            }

        tau_val = self.tau if tau is None else tau
        w_focus = self.focus_weight if focus_weight is None else focus_weight
        w_bg = self.bg_weight if bg_weight is None else bg_weight

        # Chuyển attn_weights thành danh sách nếu là Tensor
        if isinstance(attn_weights, torch.Tensor):
            layers_attn = [attn_weights[i] for i in range(attn_weights.shape[0])]
        else:
            layers_attn = attn_weights

        device = layers_attn[0].device
        num_layers = len(layers_attn)

        # Tính toán Background Mask M_bg = 1.0 - M_fg nếu có ảnh
        m_bg = None
        if image is not None:
            if spatial_shape is None:
                spatial_shape = self.infer_spatial_shape(num_visual_tokens, image=image)
            thresh = self.binary_threshold
            m_fg = self.compute_foreground_mask(image, spatial_shape=spatial_shape, threshold=thresh) # [B, 1, N_vis]
            m_bg = (1.0 - m_fg).to(device) # [B, 1, N_vis]

        total_focus_loss = torch.tensor(0.0, device=device)
        total_bg_loss = torch.tensor(0.0, device=device)
        total_vis_ratio = torch.tensor(0.0, device=device)

        for A in layers_attn:
            # A có shape [B, Q, N_mem]
            # N_mem = N_vis + 1 (qpos) + 1 (latent z)
            A_vis = A[:, :, :num_visual_tokens] # [B, Q, N_vis]
            A_vis_sum = A_vis.sum(dim=-1)       # [B, Q] - Tổng tỷ lệ chú ý vào thị giác của từng query

            # 2A. Visual Attention Ratio Loss (Hinge Loss)
            # Phạt nếu query có tỷ lệ chú ý thị giác thấp hơn tau (mặc định 0.6)
            loss_focus = F.relu(tau_val - A_vis_sum).mean()
            vis_ratio = A_vis_sum.mean()

            total_focus_loss = total_focus_loss + loss_focus
            total_vis_ratio = total_vis_ratio + vis_ratio

            # 2B. Spatial Foreground Attention Loss (Phạt attention vào background/tường)
            if m_bg is not None:
                # Phạt trọng số chú ý rơi vào các patch nền (1 - M_fg)
                bg_attn = A_vis * m_bg # [B, Q, N_vis]
                loss_bg = bg_attn.sum(dim=-1).mean()
                total_bg_loss = total_bg_loss + loss_bg

        avg_focus_loss = total_focus_loss / max(num_layers, 1)
        avg_bg_loss = total_bg_loss / max(num_layers, 1)
        avg_vis_ratio = total_vis_ratio / max(num_layers, 1)

        total_attn_loss = w_focus * avg_focus_loss + w_bg * avg_bg_loss

        return {
            "attn_loss": total_attn_loss,
            "attn_focus_loss": avg_focus_loss,
            "attn_bg_loss": avg_bg_loss,
            "attn_vis_ratio": avg_vis_ratio,
        }
