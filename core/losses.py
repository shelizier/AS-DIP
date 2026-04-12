"""Loss functions for self-supervised seismic denoising."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


def _reduce_difference(difference: torch.Tensor, mode: str = "l1") -> torch.Tensor:
    if mode == "l2":
        return difference.pow(2).mean()
    if mode == "l1":
        return difference.abs().mean()
    raise ValueError(f"Unsupported reduction mode: {mode}")


def total_variation_loss(
        image: torch.Tensor,
        mode: str = "l1",
        weight_t: float = 1.0,
        weight_x: float = 1.0,
) -> torch.Tensor:
    """Compute anisotropic total variation regularization."""
    diff_t = image[:, :, 1:, :] - image[:, :, :-1, :]
    diff_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    return weight_t * _reduce_difference(diff_t, mode=mode) + weight_x * _reduce_difference(diff_x, mode=mode)


def gradient_consistency_loss(prediction: torch.Tensor, target: torch.Tensor, mode: str = "l1") -> torch.Tensor:
    pred_t = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    return _reduce_difference(pred_t - target_t, mode=mode) + _reduce_difference(pred_x - target_x, mode=mode)


# ================= 新增：结构相似性损失 (SSIM Loss) =================
def ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """计算简易版的可导 SSIM 损失"""
    pad = window_size // 2
    # 计算局部均值
    mu_pred = F.avg_pool2d(prediction, window_size, stride=1, padding=pad)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=pad)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    # 计算局部方差与协方差
    sigma_pred_sq = F.avg_pool2d(prediction * prediction, window_size, stride=1, padding=pad) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(prediction * target, window_size, stride=1, padding=pad) - mu_pred_target

    # SSIM 稳定常数 (防止除0，根据数据范围缩放，这里假设数据经过了标准化)
    C1 = (0.01 * 2.0) ** 2
    C2 = (0.03 * 2.0) ** 2

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    # SSIM 越接近 1 越好，所以 Loss 为 1 - SSIM
    return 1.0 - ssim_map.mean()


# ================= 新增：局部正交损失 (Local Orthogonality Loss) =================
def local_orthogonality_loss(prediction: torch.Tensor, noisy_target: torch.Tensor, window: int = 9) -> torch.Tensor:
    """惩罚预测信号与残差噪声之间的局部相关性 (可导版本)"""
    residual = noisy_target - prediction
    if window % 2 == 0:
        window += 1

    mean_pred = F.avg_pool2d(prediction, kernel_size=window, stride=1, padding=window // 2)
    mean_res = F.avg_pool2d(residual, kernel_size=window, stride=1, padding=window // 2)

    centered_pred = prediction - mean_pred
    centered_res = residual - mean_res

    covariance = F.avg_pool2d(centered_pred * centered_res, kernel_size=window, stride=1, padding=window // 2)
    variance_pred = F.avg_pool2d(centered_pred.pow(2), kernel_size=window, stride=1, padding=window // 2)
    variance_res = F.avg_pool2d(centered_res.pow(2), kernel_size=window, stride=1, padding=window // 2)

    # 计算局部皮尔逊相关系数
    similarity = covariance / torch.sqrt(variance_pred * variance_res + 1e-8)

    # 我们希望相关性绝对值趋近于 0
    return similarity.abs().mean()


class CombinedLoss(nn.Module):
    """Configurable reconstruction loss with TV, gradient, SSIM, and Orthogonality regularization."""

    def __init__(
            self,
            tv_weight: float = 0.05,
            tv_mode: str = "l1",
            mse_weight: float = 1.0,
            l1_weight: float = 0.0,
            tv_weight_t: float = 1.0,
            tv_weight_x: float = 1.0,
            gradient_weight: float = 0.0,
            ssim_weight: float = 0.0,
            ortho_weight: float = 0.0,
            ortho_window: int = 9
    ) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.tv_weight = tv_weight
        self.tv_mode = tv_mode
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.tv_weight_t = tv_weight_t
        self.tv_weight_x = tv_weight_x
        self.gradient_weight = gradient_weight
        self.ssim_weight = ssim_weight
        self.ortho_weight = ortho_weight
        self.ortho_window = ortho_window

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        mse = self.mse(prediction, target)
        l1 = self.l1(prediction, target)

        # 基础重建损失
        reconstruction = self.mse_weight * mse + self.l1_weight * l1

        # 1. 结构相似性重建损失 (SSIM)
        ssim_val = torch.tensor(0.0, device=prediction.device)
        if self.ssim_weight > 0.0:
            ssim_val = ssim_loss(prediction, target)
            reconstruction += self.ssim_weight * ssim_val

        # 2. 全变分和梯度一致性
        tv = total_variation_loss(
            prediction,
            mode=self.tv_mode,
            weight_t=self.tv_weight_t,
            weight_x=self.tv_weight_x,
        )
        gradient = gradient_consistency_loss(prediction, target, mode=self.tv_mode)

        # 3. 局部正交损失 (Orthogonality)
        ortho_val = torch.tensor(0.0, device=prediction.device)
        if self.ortho_weight > 0.0:
            ortho_val = local_orthogonality_loss(prediction, target, window=self.ortho_window)

        # 总损失
        total = reconstruction + self.tv_weight * tv + self.gradient_weight * gradient + self.ortho_weight * ortho_val

        metrics = {
            "total": float(total.item()),
            "mse": float((self.mse_weight * mse).item()),
            "l1": float((self.l1_weight * l1).item()),
            "tv": float((self.tv_weight * tv).item()),
            "gradient": float((self.gradient_weight * gradient).item()),
            "ssim_loss": float((self.ssim_weight * ssim_val).item()) if self.ssim_weight > 0 else 0.0,
            "ortho_loss": float((self.ortho_weight * ortho_val).item()) if self.ortho_weight > 0 else 0.0
        }
        return total, metrics