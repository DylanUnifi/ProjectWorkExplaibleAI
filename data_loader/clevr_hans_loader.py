# data_loader/clevr_hans_loader.py

import os
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path


# Confounder metadata for CLEVR-Hans3 and CLEVR-Hans7
# Based on: https://arxiv.org/pdf/2011.12854.pdf
CLEVR_HANS_CONFOUNDERS = {
    "clevr_hans3": {
        0: {
            "confounded": True,
            "description": "Large cube + large cylinder. Confounder: large cube is always gray in train/val.",
            "attributes": ["gray", "large", "cube"],
        },
        1: {
            "confounded": True,
            "description": "Small sphere + small metal cube. Confounder: small sphere is always metal in train/val.",
            "attributes": ["metal", "small", "sphere"],
        },
        2: {
            "confounded": False,
            "description": "Large blue sphere + small yellow sphere. Not confounded.",
            "attributes": [],
        },
    },
    "clevr_hans7": {
        0: {"confounded": True, "attributes": ["gray", "large", "cube"]},
        1: {"confounded": True, "attributes": ["metal", "small", "sphere"]},
        2: {"confounded": True, "attributes": ["cube", "small", "cyan"]},
        3: {"confounded": False, "attributes": []},
        4: {"confounded": False, "attributes": []},
        5: {"confounded": False, "attributes": []},
        6: {"confounded": False, "attributes": []},
    },
}


class CLEVRHansDataset(Dataset):
    """
    CLEVR-Hans Dataset Loader.

    Supports the official download structure:
    CLEVR-Hans3/
    ├── train/
    │   ├── images/
    │   └── scenes/
    ├── val/
    │   ├── images/
    │   └── scenes/
    └── test/
        ├── images/
        └── scenes/
    """

    def __init__(
        self,
        root_dir,
        split="train",
        variant="clevr_hans3",
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

        # Load scenes — official structure: {root}/{split}/scenes/
        scenes_dir = self.root_dir / split / "scenes"
        self.scenes = []

        if scenes_dir.is_dir():
            # Official download: directory with per-image JSON or single JSON
            scene_files = sorted(scenes_dir.glob("*.json"))
            if len(scene_files) == 0:
                raise FileNotFoundError(
                    f"No JSON scene files found in {scenes_dir}. "
                    "Please check your dataset directory structure."
                )
            elif len(scene_files) == 1:
                # Single scene file for the split
                with open(scene_files[0], "r") as f:
                    data = json.load(f)
                    self.scenes = data.get("scenes", data)
            else:
                # Multiple scene files — load and merge
                for sf in scene_files:
                    with open(sf, "r") as f:
                        data = json.load(f)
                        if "scenes" in data:
                            self.scenes.extend(data["scenes"])
                        elif isinstance(data, list):
                            self.scenes.extend(data)
                        else:
                            self.scenes.append(data)
        else:
            # Fallback: old structure {root}/scenes/{split}_scenes.json
            fallback_path = self.root_dir / "scenes" / f"{split}_scenes.json"
            with open(fallback_path, "r") as f:
                data = json.load(f)
                self.scenes = data.get("scenes", data)

        # Images directory — official: {root}/{split}/images/
        self.images_dir = self.root_dir / split / "images"
        if not self.images_dir.is_dir():
            # Fallback: old structure {root}/images/{split}/
            self.images_dir = self.root_dir / "images" / split

        # Confounder info (built-in, no external file needed)
        if return_confounders:
            self.confounders = CLEVR_HANS_CONFOUNDERS.get(variant, {})

        print(f"🧩 CLEVR-Hans{self.n_classes} {split}: {len(self.scenes)} images")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]

        # Load image
        img_filename = scene.get("image_filename", f"CLEVR_Hans{self.n_classes}_{self.split}_{idx:06d}.png")
        img_path = self.images_dir / img_filename
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Class label
        label = scene.get("class_id", scene.get("class", 0))

        # Confounder info (for evaluation)
        confounder_info = {}
        if self.return_confounders and self.confounders:
            class_key = label
            if class_key in self.confounders:
                confounder_info = {
                    "is_confounded": self.confounders[class_key]["confounded"],
                    "confounder_attrs": self.confounders[class_key].get("attributes", []),
                    "objects": scene.get("objects", []),
                }

        return {
            "image": image,
            "label": label,
            "confounder_info": confounder_info,
            "image_id": img_filename,
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
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, test_loader
