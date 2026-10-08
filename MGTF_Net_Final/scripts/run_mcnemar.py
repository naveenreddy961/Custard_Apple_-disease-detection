"""McNemar test of MGTF-Net vs. baselines using REAL saved checkpoints.
Expects checkpoints in output_mgtfnet/<name>_best.pt (train baselines with train_model first)."""
import numpy as np, torch
from mgtfnet.config import Config
from mgtfnet.data import create_dataloaders
from mgtfnet.model import MGTFNet
from mgtfnet.baselines import TL_CNN, DSC_TransNet, Multi_ViT, DWTFormer, ST_CFI
from mgtfnet.stats import run_mcnemar_statistical_test

def predict(model, ckpt, loader):
    model.load_state_dict(torch.load(ckpt)); model.eval().to(Config.DEVICE)
    yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            yp += model(x.to(Config.DEVICE)).argmax(1).cpu().tolist(); yt += y.tolist()
    return np.array(yt), np.array(yp)

_, _, test_l = create_dataloaders()
d = Config.OUTPUT_DIR
y_true, y_mgtf = predict(MGTFNet(Config.NUM_CLASSES), f"{d}/MGTFNet_224x224_best.pt", test_l)
base = {n: predict(c(Config.NUM_CLASSES), f"{d}/{n}_best.pt", test_l)[1]
        for n, c in [("TL-CNN", TL_CNN), ("DSC-TransNet", DSC_TransNet), ("Multi-ViT", Multi_ViT),
                     ("DWTFormer", DWTFormer), ("ST-CFI", ST_CFI)]}
run_mcnemar_statistical_test(y_true, y_mgtf, base).to_csv(f"{d}/mcnemar_results.csv", index=False)
