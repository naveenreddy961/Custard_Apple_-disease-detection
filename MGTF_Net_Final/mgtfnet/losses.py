"""Focal loss and discriminative-LR optimizer."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config
from .model import MGTFNet

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_optimizer_and_scheduler(model, base_lr=Config.BASE_LR, epochs=Config.EPOCHS):
    if isinstance(model, MGTFNet):
        backbone_params = (
            list(model.backbone_eff.parameters()) +
            list(model.backbone_res.parameters()) +
            list(model.backbone_dense.parameters())
        )
        attn_params = (
            list(model.attn_eff.parameters()) +
            list(model.attn_res.parameters()) +
            list(model.attn_dense.parameters())
        )
        head_params = (
            list(model.proj_eff.parameters()) +
            list(model.proj_res.parameters()) +
            list(model.proj_dense.parameters()) +
            [model.pos_embedding] +
            list(model.transformer_encoder.parameters()) +
            list(model.fusion_projection.parameters()) +
            list(model.classifier.parameters())
        )

        param_groups = [
            {'params': backbone_params, 'lr': base_lr / 10.0},
            {'params': attn_params, 'lr': base_lr / 2.0},
            {'params': head_params, 'lr': base_lr}
        ]
    else:
        param_groups = [{'params': model.parameters(), 'lr': base_lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)

    # 10-Epoch Linear Warmup + Cosine Annealing
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler
