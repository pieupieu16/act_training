#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Early Stopping Module for ACT Training Pipeline
Theo dõi sự cải thiện của hàm mất mát (loss).
Dừng huấn luyện sớm khi loss bị chững lại (plateau) không giảm thêm được ít nhất min_delta sau patience epochs.
"""

from typing import Optional, Tuple


class EarlyStopping:
    """
    Trình quản lý Dừng sớm (Early Stopping) cho ACT Pipeline.

    Cơ chế hoạt động:
    - Theo dõi giá trị loss qua từng epoch.
    - Một epoch được coi là "có cải thiện" nếu: loss < (best_loss - min_delta).
    - Nếu qua `patience` epochs liên tiếp mà loss không giảm thêm được ít nhất `min_delta`,
      kích hoạt dừng sớm để tránh lãng phí tài nguyên và overfitting.
    - Hỗ trợ thêm `target_loss`: nếu loss chạm ngưỡng tuyệt đối (ví dụ < 0.0001) thì cũng dừng sớm.

    Args:
        patience (int): Số epochs liên tiếp cho phép loss không giảm thêm min_delta. Mặc định: 10.
        min_delta (float): Mức giảm tối thiểu để tính là có cải thiện đáng kể. Mặc định: 0.001.
        target_loss (float): Ngưỡng loss tuyệt đối (nếu đặt > 0 và loss < target_loss thì dừng ngay). Mặc định: 0.0 (tắt).
        metric_name (str): Tên đại lượng theo dõi ('train_loss' hoặc 'val_loss'). Mặc định: 'train_loss'.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        target_loss: float = 0.0,
        metric_name: str = "train_loss",
    ):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.target_loss = float(target_loss) if target_loss is not None else 0.0
        self.metric_name = metric_name

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.stop_reason = ""

    def step(self, current_loss: float, epoch: int) -> Tuple[bool, str]:
        """
        Kiểm tra giá trị loss tại epoch hiện tại.

        Args:
            current_loss (float): Giá trị loss của epoch hiện tại.
            epoch (int): Số thứ tự epoch hiện tại.

        Returns:
            Tuple[bool, str]: (should_stop, message)
        """
        # 1. Kiểm tra ngưỡng mục tiêu tuyệt đối (nếu được kích hoạt)
        if self.target_loss > 0 and current_loss < self.target_loss:
            self.early_stop = True
            self.stop_reason = (
                f"{self.metric_name} ({current_loss:.6f}) đã đạt/vượt ngưỡng mục tiêu tuyệt đối "
                f"(< {self.target_loss:.6f}) tại epoch {epoch}."
            )
            return True, self.stop_reason

        # 2. Kiểm tra mức cải thiện Plateau
        if current_loss < (self.best_loss - self.min_delta):
            # Có cải thiện đáng kể (giảm >= min_delta)
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.counter = 0
            return False, f"Cải thiện tốt: {self.metric_name} = {current_loss:.6f}"
        else:
            # Không cải thiện đủ mức min_delta
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                delta_achieved = self.best_loss - current_loss
                self.stop_reason = (
                    f"{self.metric_name} không giảm thêm được ít nhất {self.min_delta:.4f} "
                    f"trong {self.patience} epochs liên tiếp (Best: {self.best_loss:.6f} tại epoch {self.best_epoch}, "
                    f"Hiện tại: {current_loss:.6f}, Delta: {delta_achieved:.6f})."
                )
                return True, self.stop_reason
            else:
                return (
                    False,
                    f"Chưa cải thiện đáng kể ({self.counter}/{self.patience} epochs). "
                    f"Best: {self.best_loss:.6f} (epoch {self.best_epoch}), Hiện tại: {current_loss:.6f}",
                )

    def reset(self):
        """Đặt lại trạng thái của EarlyStopping."""
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.stop_reason = ""

