"""Train + evaluate MGTF-Net.   python scripts/train.py --img-size 224"""
import argparse
from mgtfnet.config import Config
from mgtfnet.data import create_dataloaders
from mgtfnet.model import MGTFNet
from mgtfnet.train import train_model
from mgtfnet.evaluate import plot_history, evaluate_and_plot

p = argparse.ArgumentParser()
p.add_argument("--img-size", type=int, default=224, choices=[224, 300, 380])
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--epochs", type=int, default=Config.EPOCHS)
a = p.parse_args()

train_l, val_l, test_l = create_dataloaders(a.img_size, a.batch_size)
model = MGTFNet(Config.NUM_CLASSES).to(Config.DEVICE)
name = f"MGTFNet_{a.img_size}x{a.img_size}"
hist, ckpt = train_model(model, train_l, val_l, model_name=name, epochs=a.epochs)
plot_history(hist, name)
evaluate_and_plot(model, test_l, Config.ACTIVE_CLASSES, name, weights_path=ckpt)
