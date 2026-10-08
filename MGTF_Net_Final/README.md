# MGTF-Net: Multi-backbone Gated Transformer Fusion for Mango & Custard Apple Disease Classification

PyTorch implementation of MGTF-Net (EfficientNet-B4 + ResNet-50 + DenseNet-121 backbones, CBAM/SE attention,
4-layer Transformer encoder, cross-fruit feature fusion) evaluated on **MCFD-9K** (10 classes: 5 mango + 5 custard apple).

## Repository layout
```
mgtfnet/        attention.py  model.py  ablation.py  baselines.py  losses.py
                train.py  evaluate.py  stats.py (McNemar)  tsne.py  resolution.py  data.py  config.py
scripts/        train.py  run_ablation.py  run_mcnemar.py  run_tsne.py  smoke_test.py
outputs/        figures/tables reported in the paper & response to reviewers
```

## Setup
```bash
pip install -r requirements.txt
export MCFD_DATA_DIR=/path/to/MCFD-9K   # layout: train|val|test / <class_name> / *.jpg
```

## Reproducing the reviewer-requested experiments
| Experiment | Command | Output |
|---|---|---|
| Main training + confusion matrix, ROC, kappa, class-wise report | `python scripts/train.py --img-size 224` | `outputs/confusion_metrics.png`, `roc_curves.png`, `epoch_graph.png`, `classwise_performance.png` |
| Input-resolution study (224/300/380) | run `--img-size 224`, `300`, `380` (or `mgtfnet.resolution.run_resolution_study()`) | `outputs/resolution_*.png` |
| Ablation (A: ResNet-50 → E: full MGTF-Net) | `python scripts/run_ablation.py` | `outputs/ablation_study.png` |
| Baselines (TL-CNN, DSC-TransNet, Multi-ViT, DWTFormer, ST-CFI) | `mgtfnet/baselines.py` + `train_model` | `outputs/overall_f1-scores.png` |
| McNemar test + Bonferroni correction | `python scripts/run_mcnemar.py` | `output_mgtfnet/mcnemar_results.csv` |
| t-SNE of fused latent space | `python scripts/run_tsne.py` | `outputs/tsne_visualization.png` |
| Single-crop 5-class results | set `Config.MODE = 'mango'` / `'custard'` | `outputs/mango_5class.png`, `custard_apple_5class.png` |
| Grad-CAM++ qualitative maps | see note below | `outputs/gradcam_pp.png` |

Training setup: AdamW (wd 1e-4), base LR 3e-4 (backbones ×0.1, attention ×0.5), 10-epoch warm-up + cosine,
focal loss (γ=2), 100 epochs, gradient clipping 1.0.

## Notes
- `scripts/smoke_test.py` only checks output shapes with random tensors.
- Grad-CAM++ generation code is not yet included in this repo (only the figure).
