"""t-SNE of the 1024-d fused features from a trained MGTF-Net."""
import torch
from mgtfnet.config import Config
from mgtfnet.data import create_dataloaders
from mgtfnet.model import MGTFNet
from mgtfnet.tsne import extract_latent_features, plot_tsne_latent_space

_, _, test_l = create_dataloaders()
m = MGTFNet(Config.NUM_CLASSES).to(Config.DEVICE)
m.load_state_dict(torch.load(f"{Config.OUTPUT_DIR}/MGTFNet_224x224_best.pt"))
f, y = extract_latent_features(m, test_l, Config.DEVICE)
plot_tsne_latent_space(f, y, Config.ACTIVE_CLASSES, save_path=f"{Config.OUTPUT_DIR}/tsne_feature_space.png")
