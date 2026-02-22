# data_loader/clevr_hans_loader.py

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path

class CLEVRHansDataset(Dataset):
    """
    CLEVR-Hans Dataset Loader.
    
    Structure attendue:
    clevr_hans3/ (ou clevr_hans7/)
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── scenes/
    │   ├── train_scenes.json
    │   ├── val_scenes.json
    │   └── test_scenes.json
    └── confounders.json  # Info sur confounders par classe
    """
    
    def __init__(
        self,
        root_dir,
        split="train",
        variant="clevr_hans3",  # "clevr_hans3" or "clevr_hans7"
        transform=None,
        return_confounders=True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.variant = variant
        self.transform = transform
        self.return_confounders = return_confounders
        
        # Number of classes
        self.n_classes = 3 if variant == "clevr_hans3" else 7
        
        # Load scenes
        scenes_path = self.root_dir / "scenes" / f"{split}_scenes.json"
        with open(scenes_path, "r") as f:
            data = json.load(f)
            self.scenes = data["scenes"]
        
        # Load confounder info
        if return_confounders:
            conf_path = self.root_dir / "confounders.json"
            with open(conf_path, "r") as f:
                self.confounders = json.load(f)
        
        print(f"🧩 CLEVR-Hans{self.n_classes} {split}: {len(self.scenes)} images")
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        
        # Load image
        img_path = self.root_dir / "images" / self.split / scene["image_filename"]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Class label
        label = scene["class_id"]
        
        # Confounder info (for evaluation)
        confounder_info = {}
        if self.return_confounders:
            class_name = scene["class_name"]
            confounder_info = {
                "is_confounded": self.confounders[class_name]["confounded"],
                "confounder_attrs": self.confounders[class_name].get("attributes", []),
                "objects": scene["objects"],  # Scene graph
            }
        
        return {
            "image": image,
            "label": label,
            "confounder_info": confounder_info,
            "image_id": scene["image_filename"],
        }


def get_clevr_hans_loaders(root_dir, variant="clevr_hans3", batch_size=64, num_workers=8):
    """Get train/val/test loaders."""
    from torchvision import transforms as T
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = CLEVRHansDataset(root_dir, split="train", variant=variant, transform=transform)
    val_dataset = CLEVRHansDataset(root_dir, split="val", variant=variant, transform=transform)
    test_dataset = CLEVRHansDataset(root_dir, split="test", variant=variant, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
