"""Proposed MGTF-Net architecture."""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights, ResNet50_Weights, DenseNet121_Weights
from .config import Config
from .attention import CBAM, SEBlock

class MGTFNet(nn.Module):
    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        d_model=512,
        nhead=8,
        num_layers=4,
        ffn_dim=2048,
        target_spatial_dim=7
    ):
        super().__init__()
        self.target_spatial_dim = target_spatial_dim
        num_patches = target_spatial_dim * target_spatial_dim  # 49
        total_tokens = 3 * num_patches                        # 147

        # 1. Multi-Scale Pre-trained Backbones
        effnet = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        self.backbone_eff = effnet.features  # 1792 channels

        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone_res = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )  # 2048 channels

        densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.backbone_dense = densenet.features  # 1024 channels

        # Adaptive pooling guarantees 7x7 spatial maps across resolutions (224, 300, 380)
        self.spatial_pool = nn.AdaptiveAvgPool2d((target_spatial_dim, target_spatial_dim))

        # 2. Architecture-Matched Attention
        self.attn_eff = CBAM(1792, ratio=16)
        self.attn_res = SEBlock(2048, ratio=16)
        self.attn_dense = CBAM(1024, ratio=16)

        # 3. Token Linear Projections and Learnable Positional Embeddings
        self.proj_eff = nn.Linear(1792, d_model)
        self.proj_res = nn.Linear(2048, d_model)
        self.proj_dense = nn.Linear(1024, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, total_tokens, d_model) * 0.02)

        # 4. Multi-Head Self-Attention Transformer Encoder (L=4, h=8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Cross-Fruit Feature Fusion Module (5376 -> 1024)
        self.gap = nn.AdaptiveAvgPool2d(1)
        fusion_dim = d_model + 1792 + 2048 + 1024  # 5376
        self.fusion_projection = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

        # 6. Two-stage Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        b = x.size(0)

        # Feature Extraction
        f_e = self.spatial_pool(self.backbone_eff(x))
        f_r = self.spatial_pool(self.backbone_res(x))
        f_d = self.spatial_pool(self.backbone_dense(x))

        # Attention Enhancement
        f_e_attn = self.attn_eff(f_e)
        f_r_attn = self.attn_res(f_r)
        f_d_attn = self.attn_dense(f_d)

        # Patch Token Flattening: (B, C, 7, 7) -> (B, 49, C)
        t_e = f_e_attn.flatten(2).permute(0, 2, 1)
        t_r = f_r_attn.flatten(2).permute(0, 2, 1)
        t_d = f_d_attn.flatten(2).permute(0, 2, 1)

        t_e = self.proj_eff(t_e)
        t_r = self.proj_res(t_r)
        t_d = self.proj_dense(t_d)

        # Concatenate Tokens along sequence axis (147 tokens)
        t_tokens = torch.cat([t_e, t_r, t_d], dim=1) + self.pos_embedding
        t_out = self.transformer_encoder(t_tokens)
        g = torch.mean(t_out, dim=1)  # Mean pool over tokens: (B, 512)

        # Global Average Pooling on feature maps
        p_e = self.gap(f_e_attn).view(b, -1)
        p_r = self.gap(f_r_attn).view(b, -1)
        p_d = self.gap(f_d_attn).view(b, -1)

        # Composite Fusion
        g_fuse = torch.cat([g, p_e, p_r, p_d], dim=1)  # (B, 5376)
        g_prime = self.fusion_projection(g_fuse)       # (B, 1024)

        logits = self.classifier(g_prime)
        return logits   
