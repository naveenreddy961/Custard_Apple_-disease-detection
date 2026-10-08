"""Train each ablation configuration (A-D) and report test accuracy / macro-F1."""
import torch
from sklearn.metrics import accuracy_score, f1_score
from mgtfnet.config import Config
from mgtfnet.data import create_dataloaders
from mgtfnet.ablation import (ConfigA_ResNet50, ConfigB_ResNet50_CBAM,
                              ConfigC_ResNet50_CBAM_Transformer, ConfigD_MultiBB_NoTransformer)
from mgtfnet.model import MGTFNet
from mgtfnet.train import train_model

configs = {"A_ResNet50": ConfigA_ResNet50, "B_ResNet50_CBAM": ConfigB_ResNet50_CBAM,
           "C_CBAM_Transformer": ConfigC_ResNet50_CBAM_Transformer,
           "D_MultiBB_NoTransformer": ConfigD_MultiBB_NoTransformer, "E_Full_MGTFNet": MGTFNet}
train_l, val_l, test_l = create_dataloaders()
for name, cls in configs.items():
    m = cls(Config.NUM_CLASSES).to(Config.DEVICE)
    _, ckpt = train_model(m, train_l, val_l, model_name=name)
    m.load_state_dict(torch.load(ckpt)); m.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, y in test_l:
            yp += m(x.to(Config.DEVICE)).argmax(1).cpu().tolist(); yt += y.tolist()
    print(f"{name}: acc={100*accuracy_score(yt,yp):.2f}  macroF1={100*f1_score(yt,yp,average='macro'):.2f}")
