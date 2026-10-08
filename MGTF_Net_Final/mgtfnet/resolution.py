"""Input-resolution study (224 / 300 / 380)."""
from .config import Config
from .data import create_dataloaders
from .model import MGTFNet
from .train import train_model
from .evaluate import plot_history, evaluate_and_plot

def run_resolution_study(resolutions=(224, 300, 380)):
    
    for res in resolutions:
        print(f"\n=======================================================")
        print(f"  Training MGTF-Net with Resolution: {res} x {res}")
        print(f"=======================================================")
        
        # Adjust batch size for memory stability on 380x380 if needed
        batch_size = 16 if res == 380 else 32
        train_loader, val_loader, test_loader = create_dataloaders(img_size=res, batch_size=batch_size)
        
        model = MGTFNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
        history, best_ckpt = train_model(model, train_loader, val_loader, model_name=f"MGTFNet_{res}x{res}", epochs=Config.EPOCHS)
        
        plot_history(history, model_name=f"MGTFNet_{res}x{res}")
        evaluate_and_plot(model, test_loader, Config.ACTIVE_CLASSES, model_name=f"MGTFNet_{res}x{res}", weights_path=best_ckpt)