"""Augmentation, dataset and dataloaders."""
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .config import Config

def get_transforms(img_size):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.10, 0.10),
            scale=(0.85, 1.15)
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform

class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.samples = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        self.transform = transform
        
        for cls_name in class_names:
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_folder, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def create_dataloaders(img_size=Config.IMG_SIZE, batch_size=Config.BATCH_SIZE):
    train_tf, eval_tf = get_transforms(img_size)
    
    train_dataset = PlantDiseaseDataset(Config.TRAIN_DIR, Config.ACTIVE_CLASSES, transform=train_tf)
    val_dataset = PlantDiseaseDataset(Config.VAL_DIR, Config.ACTIVE_CLASSES, transform=eval_tf)
    test_dataset = PlantDiseaseDataset(Config.TEST_DIR, Config.ACTIVE_CLASSES, transform=eval_tf)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, test_loader