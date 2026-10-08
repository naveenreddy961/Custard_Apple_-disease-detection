"""Ablation configurations A-D (Full MGTF-Net = configuration E)."""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights, ResNet50_Weights, DenseNet121_Weights
from .config import Config
from .attention import CBAM, SEBlock

class ConfigA_ResNet50(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.gap(self.backbone(x)).flatten(1)
        return self.fc(feat)

class ConfigB_ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.cbam = CBAM(2048, ratio=16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.gap(self.cbam(self.backbone(x))).flatten(1)
        return self.fc(feat)

class ConfigC_ResNet50_CBAM_Transformer(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, d_model=512):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.cbam = CBAM(2048, ratio=16)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.proj = nn.Linear(2048, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 49, d_model) * 0.02)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=2048,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        feat = self.pool(self.cbam(self.backbone(x)))
        tokens = feat.flatten(2).permute(0, 2, 1)
        tokens = self.proj(tokens) + self.pos_emb
        t_out = self.transformer(tokens)
        g = torch.mean(t_out, dim=1)
        return self.fc(g)

class ConfigD_MultiBB_NoTransformer(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super().__init__()
        eff = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        self.backbone_eff = eff.features
        res = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone_res = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool,
            res.layer1, res.layer2, res.layer3, res.layer4
        )
        dense = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.backbone_dense = dense.features

        self.attn_eff = CBAM(1792, ratio=16)
        self.attn_res = SEBlock(2048, ratio=16)
        self.attn_dense = CBAM(1024, ratio=16)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(1792 + 2048 + 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        b = x.size(0)
        p_e = self.gap(self.attn_eff(self.backbone_eff(x))).view(b, -1)
        p_r = self.gap(self.attn_res(self.backbone_res(x))).view(b, -1)
        p_d = self.gap(self.attn_dense(self.backbone_dense(x))).view(b, -1)
        return self.classifier(torch.cat([p_e, p_r, p_d], dim=1))
