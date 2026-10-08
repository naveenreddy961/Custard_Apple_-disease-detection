"""Re-implemented baselines: TL-CNN, DSC-TransNet, Multi-ViT, DWTFormer, ST-CFI."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights, MobileNet_V3_Small_Weights, Swin_T_Weights, ViT_B_16_Weights

# ==============================================================================
# BASELINE 1: TL-CNN (Transfer Learning CNN - Singh et al.)
# Standard transfer learning using pre-trained deep residual network.
# ==============================================================================
class TL_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.gap(self.features(x)).flatten(1)
        return self.classifier(feat)


# ==============================================================================
# BASELINE 2: DSC-TransNet (Depthwise Separable Conv + Transformer - Mathew et al.)
# Combines lightweight depthwise separable convolutions with a Transformer encoder.
# ==============================================================================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class DSC_TransNet(nn.Module):
    def __init__(self, num_classes=10, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.stem = mobilenet.features[:4]  # Early feature representations
        
        self.dsc1 = DepthwiseSeparableConv(24, 64, stride=2)
        self.dsc2 = DepthwiseSeparableConv(64, 128, stride=2)
        self.dsc3 = DepthwiseSeparableConv(128, d_model, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.pos_emb = nn.Parameter(torch.randn(1, 49, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.pool(self.dsc3(x))
        tokens = x.flatten(2).permute(0, 2, 1) + self.pos_emb
        t_out = self.transformer(tokens)
        return self.classifier(torch.mean(t_out, dim=1))


# ==============================================================================
# BASELINE 3: Multi-ViT (Multi-Vision Transformer - Baek et al.)
# Pretrained ViT backbone extracting and aggregating multi-stage attention representations.
# ==============================================================================
class Multi_ViT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder = vit.encoder
        hidden_dim = vit.heads[0].in_features  # 768

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        n = x.shape[0]
        # Project patches to embedding
        x = self.conv_proj(x).flatten(2).transpose(1, 2)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        cls_token = x[:, 0]
        return self.head(cls_token)


# ==============================================================================
# BASELINE 4: DWTFormer (Discrete Wavelet + Transformer Fusion - Xiang et al.)
# Emulates 2D Haar Wavelet Decomposition (LL, LH, HL, HH subbands) fused with spatial tokens.
# ==============================================================================
class HaarWaveletDecomposition2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Sub-sampling components
        ll = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) * 0.5
        lh = (-x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) * 0.5
        hl = (-x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) * 0.5
        hh = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) * 0.5
        return torch.cat([ll, lh, hl, hh], dim=1)  # 12 frequency channels for RGB

class DWTFormer(nn.Module):
    def __init__(self, num_classes=10, d_model=384, nhead=6, num_layers=4):
        super().__init__()
        self.dwt = HaarWaveletDecomposition2D()
        
        # Frequency domain branch (12 channels)
        self.freq_conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Spatial domain branch
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.pos_emb = nn.Parameter(torch.randn(1, 49, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        freq_feat = self.freq_conv(self.dwt(x))
        spat_feat = self.spatial_conv(x)
        fused_map = torch.cat([freq_feat, spat_feat], dim=1)  # d_model channels
        
        tokens = fused_map.flatten(2).permute(0, 2, 1) + self.pos_emb
        t_out = self.transformer(tokens)
        return self.classifier(torch.mean(t_out, dim=1))


# ==============================================================================
# BASELINE 5: ST-CFI (Swin Transformer + Conv Feature Interactions - Yu et al.)
# Pretrained Swin Transformer integrated with local depthwise convolutional bypass blocks.
# ==============================================================================
class ST_CFI(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        swin = models.swin_t(weights=Swin_T_Weights.DEFAULT)
        self.features = swin.features
        self.norm = swin.norm
        embed_dim = swin.head.in_features  # 768
        
        # Convolutional Feature Interaction (CFI) residual block
        self.cfi_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Swin features output layout: (B, H, W, C)
        feat = self.features(x)
        feat = self.norm(feat).permute(0, 3, 1, 2)  # Layout: (B, C, H, W)
        
        # Local interaction refinement
        feat = feat + self.cfi_conv(feat)
        out = self.gap(feat).flatten(1)
        return self.classifier(out)
