"""Hyperparameters, class definitions and paths."""
import os
import math
import copy
import numpy as np
import torch

class Config:
    # Resolutions to evaluate: 224, 300, 380
    IMG_SIZE = 224  
    BATCH_SIZE = 32
    EPOCHS = 100
    BASE_LR = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mode selection: 'joint' (10 classes), 'mango' (5 classes), 'custard' (5 classes)
    MODE = 'joint' 
    
    # Class definitions (5 Mango + 5 Custard Apple = 10 Classes)
    CLASSES_MANGO = [
        "Mango Anthracnose",
        "Mango Powdery Mildew",
        "Mango Bacterial Canker",
        "Mango Die-Back",
        "Mango Healthy"
    ]
    
    CLASSES_CUSTARD = [
        "Custard Anthracnose",
        "Custard Mealy Bug",
        "Custard Black Canker",
        "Custard Leaf Spot",
        "Custard Healthy"
    ]
    
    CLASSES_JOINT = CLASSES_MANGO + CLASSES_CUSTARD
    
    # Select active class names based on mode
    if MODE == 'mango':
        ACTIVE_CLASSES = CLASSES_MANGO
    elif MODE == 'custard':
        ACTIVE_CLASSES = CLASSES_CUSTARD
    else:
        ACTIVE_CLASSES = CLASSES_JOINT
        
    NUM_CLASSES = len(ACTIVE_CLASSES)

    # Set your folder paths (Structure: DATA_DIR/train/<class_name>/*.jpg)
    DATA_DIR = os.environ.get("MCFD_DATA_DIR", "./data/MCFD-9K")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    OUTPUT_DIR = "./output_mgtfnet"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
