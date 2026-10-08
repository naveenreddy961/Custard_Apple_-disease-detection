"""Forward-pass shape check ONLY (random tensors, no results). Safe to run without data."""
import torch
from mgtfnet.model import MGTFNet
from mgtfnet.baselines import TL_CNN, DSC_TransNet, Multi_ViT, DWTFormer, ST_CFI
x = torch.randn(2, 3, 224, 224)
for n, m in [("MGTF-Net", MGTFNet(10)), ("TL-CNN", TL_CNN()), ("DSC-TransNet", DSC_TransNet()),
             ("Multi-ViT", Multi_ViT()), ("DWTFormer", DWTFormer()), ("ST-CFI", ST_CFI())]:
    m.eval(); print(f"{n:<14}", tuple(m(x).shape))
