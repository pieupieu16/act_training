#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Backbone for ACT:
Độc quyền sử dụng ConvNeXt-Nano (Pure-CNN SOTA) cho dữ liệu RGB-D 4 kênh.
- Nhận diện không gian 3D và biên cạnh vật thể sắc nét với kernel 7x7 và Inverted Bottleneck.
- 1x1 Conv Image Projection 640 -> d_model (512).
- 2D Sinusoidal Positional Encoding (Fixed).
- Không có bất kỳ nhánh rẽ fallback sang ResNet; nếu thiếu ConvNeXt-Nano sẽ lập tức báo lỗi.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding2D(nn.Module):
    """
    2D Sinusoidal Positional Encoding cố định (không học - Fixed)
    Mã hóa tọa độ không gian 2D cho từng patch đặc trưng theo công thức DETR/ACT.
    """
    def __init__(self, d_model: int = 512, temperature: float = 10000.0):
        super().__init__()
        assert d_model % 4 == 0, f"d_model ({d_model}) phải chia hết cho 4 để chia đều cho sin/cos x và y."
        self.d_model = d_model
        self.temperature = temperature
        self.num_pos_feats = d_model // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: pos: [B, d_model, H, W]
        """
        B, _, H, W = x.shape
        device = x.device
        
        # Tạo lưới tọa độ 1..H và 1..W
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(H, 1)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        # Ghép [H, W, d_model] -> Permute [d_model, H, W] -> Repeat cho Batch [B, d_model, H, W]
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        return pos


class RGBDConvNeXtBackbone(nn.Module):
    """
    Khối thị giác RGB-D sử dụng ConvNeXt-Nano cho mô hình ACT:
    - Nhận vào ảnh 4 kênh [B, 4, H, W] (3 RGB + 1 Depth).
    - Sử dụng ConvNeXt-Nano (timm) với in_chans=4, kernel lớn 7x7 và Inverted Bottleneck
      giúp định vị hình học và biên vật thể trong không gian 3D vượt trội.
    - 1x1 Conv Image Projection: Chiếu 640 kênh đặc trưng về d_model (512).
    - 2D Sinusoidal PE: Mã hóa tọa độ không gian 2D cho từng patch.
    - Đầu ra: visual_tokens [B, N_patches, d_model] ([B, 300, 512] với ảnh 640x480).
    """
    def __init__(
        self,
        d_model: int = 512,
        freeze_backbone: bool = True,
        in_channels: int = 4,
        model_name: str = "convnext_nano",
        pretrained: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.model_name = model_name

        # Kiểm tra bắt buộc thư viện timm
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "[LỖI BẮT BUỘC] Không tìm thấy thư viện 'timm' để khởi tạo ConvNeXt-Nano! "
                "Hệ thống không hỗ trợ ResNet-18. Vui lòng cài đặt: pip install timm>=1.0.0"
            ) from exc

        # Khởi tạo ConvNeXt-Nano từ timm
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_channels,
            )
            in_features = self.backbone.num_features
        except Exception as exc:
            raise RuntimeError(
                f"[LỖI BẮT BUỘC] Không thể tải/khởi tạo mô hình '{model_name}' với {in_channels} kênh: {exc}. "
                f"Kiểm tra lại kết nối mạng hoặc trọng số cache của timm."
            ) from exc

        # Đóng băng backbone nếu freeze_backbone = True
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # 1x1 Conv Image Projection (Trainable)
        self.img_proj = nn.Conv2d(in_features, d_model, kernel_size=1)

        # 2D Sinusoidal Positional Encoding (Fixed)
        self.pos_embed_2d = SinusoidalPositionEmbedding2D(d_model=d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_feature_map: bool = False,
    ):
        """
        Input:
            x: [B, 4, H, W] - Ảnh 4 kênh RGB-D đã chuẩn hóa
            return_feature_map: Nếu True, trả về cả tuple (visual_tokens, proj_feat)
        Output:
            visual_tokens: [B, N_patches, d_model]
            (Tùy chọn) proj_feat: [B, d_model, H_feat, W_feat] nếu return_feature_map=True
        """
        # Trích xuất đặc trưng ConvNeXt-Nano
        feat = self.backbone.forward_features(x)  # [B, 640, H_feat, W_feat]
        
        # Chiếu về d_model và thêm mã hóa vị trí 2D
        proj_feat = self.img_proj(feat)           # [B, d_model, H_feat, W_feat]
        pos = self.pos_embed_2d(proj_feat)         # [B, d_model, H_feat, W_feat]
        feat_with_pos = proj_feat + pos

        # Làm phẳng thành chuỗi Visual Tokens cho Transformer Decoder
        visual_tokens = feat_with_pos.flatten(2).permute(0, 2, 1)  # [B, N_patches, d_model]
        if return_feature_map:
            return visual_tokens, proj_feat
        return visual_tokens


def build_vision_backbone(
    backbone_type: str = "convnext_nano",
    d_model: int = 512,
    freeze_backbone: bool = True,
    in_channels: int = 4,
) -> nn.Module:
    """
    Factory tạo Vision Backbone cho ACT.
    Độc quyền hỗ trợ ConvNeXt-Nano. Mọi yêu cầu kiến trúc khác (bao gồm resnet18) đều lập tức báo lỗi.
    """
    if backbone_type != "convnext_nano":
        raise ValueError(
            f"[LỖI NGHIÊM TRỌNG] Dự án đã loại bỏ hoàn toàn ResNet-18 và chỉ chấp nhận duy nhất 'convnext_nano'. "
            f"Giá trị cung cấp không hợp lệ: '{backbone_type}'."
        )

    return RGBDConvNeXtBackbone(
        d_model=d_model,
        freeze_backbone=freeze_backbone,
        in_channels=in_channels,
        model_name="convnext_nano",
    )
