# data_loader/utils.py
from torchvision import transforms
from torch.utils.data import Subset
import torch
import numpy as np
import numpy as np

def create_clevr_mask(objects, target_attrs=None, target_ids=None, image_size=(240, 320)):
    """
    Create a binary mask for objects matching target_attrs or target_ids.
    If both are None, creates a mask for all objects.
    """
    mask = np.zeros(image_size, dtype=np.float32)
    
    for i, obj in enumerate(objects):
        if target_ids is not None and i not in target_ids:
            continue
            
        if target_attrs is not None and len(target_attrs) > 0:
            obj_attrs = [obj.get("color"), obj.get("size"), obj.get("shape"), obj.get("material")]
            if not all(attr in obj_attrs for attr in target_attrs if attr is not None):
                continue
                
        if "pixel_coords" in obj:
            coords = obj["pixel_coords"]
            col, row = int(coords[0]), int(coords[1])
            radius = 35 if obj.get("size") == "large" else 20
            
            Y, X = np.ogrid[:image_size[0], :image_size[1]]
            dist_from_center = np.sqrt((X - col)**2 + (Y - row)**2)
            mask[dist_from_center <= radius] = 1.0
            
    return mask


def build_transform(grayscale: bool = True, augment: bool = False):
    """Standardized transformation pipeline."""
    tfms = []
    if grayscale:
        tfms.append(transforms.Grayscale())

    if augment:
        tfms.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(28, padding=4),
        ])

    tfms.append(transforms.ToTensor())
    if grayscale:
        tfms.append(transforms.Normalize(mean=[0.5], std=[0.5]))  # 1 channel
    else:
        tfms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5]))     # 3 channels

    return transforms.Compose(tfms)


def relabel_subset(subset: Subset, targets, binary_classes):
    """
    Map the second class (binary_classes[1]) to 1, the other to 0 in `targets` in-place.
    """
    binary_targets = [1 if int(targets[i]) == int(binary_classes[1]) else 0 for i in subset.indices]
    for idx, label in zip(subset.indices, binary_targets):
        targets[idx] = int(label)
    return subset




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

class CLE4EVRDataset(Dataset):
    """
    CLE4EVR Dataset Loader.
    Images are 240x320.
    Binary classification: positive if exists two objects with same color and shape.
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.n_classes = 2

        scenes_dir = self.root_dir / split / "scenes"
        self.scenes = []

        if scenes_dir.is_dir():
            scene_files = sorted(scenes_dir.glob("*.json"))
            aggregated = [sf for sf in scene_files if "scenes" in sf.stem and "_classid_" not in sf.stem]
            files_to_load = aggregated if aggregated else scene_files
            if len(files_to_load) == 1:
                with open(files_to_load[0], "r") as f:
                    data = json.load(f)
                    self.scenes = data.get("scenes", data)
            else:
                for sf in files_to_load:
                    with open(sf, "r") as f:
                        data = json.load(f)
                        if "scenes" in data:
                            self.scenes.extend(data["scenes"])
                        elif isinstance(data, list):
                            self.scenes.extend(data)
                        else:
                            self.scenes.append(data)
        else:
            fallback_path = self.root_dir / "scenes" / f"{split}_scenes.json"
            if fallback_path.exists():
                with open(fallback_path, "r") as f:
                    data = json.load(f)
                    self.scenes = data.get("scenes", data)

        self.images_dir = self.root_dir / split / "images"
        if not self.images_dir.is_dir():
            self.images_dir = self.root_dir / "images" / split

        print(f"CLE4EVR {split}: {len(self.scenes)} images")

    def __len__(self):
        return len(self.scenes)

    def _compute_label_and_targets(self, objects):
        # positive iff at least two objects have the same color and shape
        target_ids = []
        label = 0
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if objects[i].get("color") == objects[j].get("color") and objects[i].get("shape") == objects[j].get("shape"):
                    target_ids.extend([i, j])
                    label = 1
        return label, list(set(target_ids))

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        img_filename = scene.get("image_filename", f"CLE4EVR_{self.split}_{idx:06d}.png")
        img_path = self.images_dir / img_filename
        
        # Fallback if image has a different name
        if not img_path.exists():
            img_filename = scene.get("image_filename", f"CLEVR_HANS_{self.split}_{idx:06d}.png")
            img_path = self.images_dir / img_filename

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Force compute CLE4EVR rule (ignore CLEVR-Hans class_id if present)
        label, target_ids = self._compute_label_and_targets(scene.get("objects", []))
        
        gt_mask = create_clevr_mask(scene.get("objects", []), target_ids=target_ids if label == 1 else None)
        gt_mask = torch.from_numpy(gt_mask).float()
        conf_mask = torch.zeros_like(gt_mask)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "ground_truth_mask": gt_mask,
            "confounder_mask": conf_mask,
            "confounder_info": {}
        }



class CLEVRHansDataset(Dataset):
    """
    CLEVR-Hans Dataset Loader.

    Supports the official download structure:
    CLEVR-Hans3/
    +-- train/
    |   +-- images/
    |   +-- scenes/
    +-- val/
    |   +-- images/
    |   +-- scenes/
    +-- test/
        +-- images/
        +-- scenes/
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

        # Load scenes -- official structure: {root}/{split}/scenes/
        scenes_dir = self.root_dir / split / "scenes"
        self.scenes = []

        if scenes_dir.is_dir():
            scene_files = sorted(scenes_dir.glob("*.json"))
            # Prefer aggregated scene files (e.g. CLEVR_HANS_scenes_train.json)
            # to avoid double-counting when both aggregated and individual per-class
            # files (e.g. CLEVR_HANS_scenes_train_classid_0.json) coexist.
            aggregated = [sf for sf in scene_files if "scenes" in sf.stem and "_classid_" not in sf.stem]
            files_to_load = aggregated if aggregated else scene_files
            if len(files_to_load) == 1:
                with open(files_to_load[0], "r") as f:
                    data = json.load(f)
                    self.scenes = data.get("scenes", data)
            else:
                for sf in files_to_load:
                    with open(sf, "r") as f:
                        data = json.load(f)
                        if "scenes" in data:
                            self.scenes.extend(data["scenes"])
                        elif isinstance(data, list):
                            self.scenes.extend(data)
                        else:
                            self.scenes.append(data)
        else:
            fallback_path = self.root_dir / "scenes" / f"{split}_scenes.json"
            with open(fallback_path, "r") as f:
                data = json.load(f)
                self.scenes = data.get("scenes", data)

        # Images directory
        self.images_dir = self.root_dir / split / "images"
        if not self.images_dir.is_dir():
            self.images_dir = self.root_dir / "images" / split

        # Confounder info (built-in)
        if return_confounders:
            self.confounders = CLEVR_HANS_CONFOUNDERS.get(variant, {})

        print(f"CLEVR-Hans{self.n_classes} {split}: {len(self.scenes)} images")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]

        img_filename = scene.get("image_filename", f"CLEVR_Hans3_{self.split}_{idx:06d}.png")
        img_path = self.images_dir / img_filename
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = scene.get("class_id", scene.get("class", 0))

        confounder_info = {}
        is_confounded = False
        
        objects = scene.get("objects", [])
        conf_mask = torch.zeros((240, 320), dtype=torch.float32)
        
        if self.return_confounders and self.confounders:
            class_key = label
            if class_key in self.confounders:
                is_confounded = self.confounders[class_key]["confounded"]
                conf_attrs = self.confounders[class_key].get("attributes", [])
                confounder_info = {
                    "is_confounded": is_confounded,
                    "confounder_attrs": conf_attrs,
                }
                if is_confounded and len(conf_attrs) > 0:
                    conf_mask = torch.from_numpy(create_clevr_mask(objects, target_attrs=conf_attrs)).float()

        gt_mask = torch.from_numpy(create_clevr_mask(objects)).float()

        # Group index for GroupDRO: 0=confounded, 1=clean
        group_idx = 0 if is_confounded else 1

        return {
            "image": image,
            "label": label,
            "confounder_info": confounder_info,
            "image_id": img_filename,
            "group_idx": group_idx,
            "ground_truth_mask": gt_mask,
            "confounder_mask": conf_mask,
        }


def clevr_collate_fn(batch):
    """Custom collate that handles variable-length confounder_info."""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    image_ids = [item.get("image_id", "") for item in batch]
    confounder_infos = [item.get("confounder_info", {}) for item in batch]
    group_idx = torch.tensor([item.get("group_idx", 0) for item in batch], dtype=torch.long)
    
    gt_masks = torch.stack([item.get("ground_truth_mask", torch.zeros((224, 224))) for item in batch])
    conf_masks = torch.stack([item.get("confounder_mask", torch.zeros((224, 224))) for item in batch])

    return {
        "image": images,
        "label": labels,
        "confounder_info": confounder_infos,
        "image_id": image_ids,
        "group_idx": group_idx,
        "ground_truth_mask": gt_masks,
        "confounder_mask": conf_masks,
    }


def get_clevr_hans_loaders(
    root_dir,
    variant="clevr_hans3",
    batch_size=64,
    num_workers=16,
    train_augmentation=False,
    max_samples=None,
):
    """Get train/val/test loaders."""
    from torchvision import transforms as T

    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if train_augmentation:
        train_transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
        ])
    else:
        train_transform = eval_transform

    train_dataset = CLEVRHansDataset(root_dir, split="train", variant=variant, transform=train_transform)
    val_dataset = CLEVRHansDataset(root_dir, split="val", variant=variant, transform=eval_transform)
    test_dataset = CLEVRHansDataset(root_dir, split="test", variant=variant, transform=eval_transform)

    if max_samples is not None:
        from torch.utils.data import Subset
        import numpy as np
        train_dataset = Subset(train_dataset, np.arange(min(max_samples, len(train_dataset))))
        val_dataset = Subset(val_dataset, np.arange(min(max_samples, len(val_dataset))))
        test_dataset = Subset(test_dataset, np.arange(min(max_samples, len(test_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=clevr_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=clevr_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=clevr_collate_fn,
    )

    return train_loader, val_loader, test_loader


# data_loader/mnmath_loader.py

import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Note: We avoid importing joblib at the module level if it's not installed yet.
# It will be imported in __init__ if needed, but standard JSON or numpy loading 
# is preferred if joblib isn't available.

class MNMathDataset(Dataset):
    """
    MNMath Dataset Loader.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load joblib if available
        try:
            import joblib
            self.joblib = joblib
        except ImportError:
            raise ImportError("Please install joblib (`pip install joblib`) to load MNMath dataset.")

        split_dir = self.root_dir / split
        
        self.list_images = glob.glob(os.path.join(split_dir, "*.png"))
        self.list_images = sorted(self.list_images, key=self._extract_number)
        
        self.labels = []
        self.concepts = []
        
        new_images = self.list_images.copy()
        
        for item in self.list_images:
            name = os.path.splitext(os.path.basename(item))[0]
            meta_id = name.split("_")[-1]
            
            meta_scene = split_dir / f"{meta_id}.joblib"
            
            if not meta_scene.exists():
                new_images.remove(item)
                continue
                
            data = self.joblib.load(meta_scene)
            
            # The label might be an array; we take the first element or interpret it
            # depending on the task. Typically MNMath returns a binary value or a digit.
            # Assuming labels is a list of bools, we take the primary label (e.g. data["label"][0]).
            label_data = data["label"]
            if isinstance(label_data, list) or isinstance(label_data, np.ndarray):
                label_vec = np.array(label_data).astype(np.int64)
            else:
                label_vec = np.array([label_data]).astype(np.int64)
                
            self.labels.append(label_vec)
            
            concept_values = data["meta"]["concepts"]
            concepts = np.array(concept_values).astype(np.int64).flatten()
            self.concepts.append(concepts)
            
        self.list_images = new_images
        
        self.num_equations = len(self.labels[0]) if len(self.labels) > 0 else 0
        self.num_concepts = len(self.concepts[0]) if len(self.concepts) > 0 else 0
        
        print(f"MNMath {split}: {len(self.list_images)} images loaded.")
        
    def _extract_number(self, path):
        match = re.search(r"\d+", path)
        return int(match.group()) if match else 0

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        img_path = self.list_images[idx]
        image = Image.open(img_path).convert("L")  # Grayscale
        
        image_tensor = self.transform(image) if self.transform else image
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        concepts = torch.tensor(self.concepts[idx], dtype=torch.long)
        
        # Ground truth mask for MNMath: non-zero pixels (bright digits on dark background)
        gt_mask = (torch.tensor(np.array(image)).float() > 0.1).float()
        conf_mask = torch.zeros_like(gt_mask)
        
        # Format identical to CLEVR-Hans for compatibility with train_all_models.py
        return {
            "image": image_tensor if 'image_tensor' in locals() else (self.transform(image) if self.transform else image),
            "label": label,
            "concepts": concepts,
            "group": 0,  # default
            "scene": {"concepts": concepts.tolist()},
            "confounder_info": {},
            "image_path": str(img_path),
            "ground_truth_mask": gt_mask,
            "confounder_mask": conf_mask,
        }


def get_mnmath_loaders(
    root_dir="./data/mnmath",
    batch_size=32,
    num_workers=16,
    pin_memory=True,
    image_size=(32, 128), # typical size for MNMath concatenation
    max_samples=None,
):
    """
    Returns train, val, test dataloaders for the MNMath dataset.
    """
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    transform_eval = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # In case data doesn't exist yet, we don't crash unless we actually load it.
    try:
        train_dataset = MNMathDataset(root_dir, split="train", transform=transform_train)
        val_dataset = MNMathDataset(root_dir, split="val", transform=transform_eval)
        test_dataset = MNMathDataset(root_dir, split="test", transform=transform_eval)
        
        if max_samples is not None:
            from torch.utils.data import Subset
            import numpy as np
            train_dataset = Subset(train_dataset, np.arange(min(max_samples, len(train_dataset))))
            val_dataset = Subset(val_dataset, np.arange(min(max_samples, len(val_dataset))))
            test_dataset = Subset(test_dataset, np.arange(min(max_samples, len(test_dataset))))
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=(max_samples is None),
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        )
        num_equations = train_dataset.num_equations if hasattr(train_dataset, "num_equations") else train_dataset.dataset.num_equations
        num_concepts = train_dataset.num_concepts if hasattr(train_dataset, "num_concepts") else train_dataset.dataset.num_concepts
        return train_loader, val_loader, test_loader, num_equations, num_concepts
    except Exception as e:
        print(f"Warning: MNMath dataset not found or missing dependency: {e}. Returning empty loaders.")
        return None, None, None, 0, 0

def get_cle4evr_loaders(
    root_dir,
    batch_size=64,
    num_workers=16,
    max_samples=None,
):
    """Get train/val/test loaders for CLE4EVR."""
    from torchvision import transforms as T
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transform = eval_transform

    train_dataset = CLE4EVRDataset(root_dir, split="train", transform=train_transform)
    val_dataset = CLE4EVRDataset(root_dir, split="val", transform=eval_transform)
    test_dataset = CLE4EVRDataset(root_dir, split="test", transform=eval_transform)

    if max_samples is not None:
        train_dataset = Subset(train_dataset, np.arange(min(max_samples, len(train_dataset))))
        val_dataset = Subset(val_dataset, np.arange(min(max_samples, len(val_dataset))))
        test_dataset = Subset(test_dataset, np.arange(min(max_samples, len(test_dataset))))

    # Using standard default collate since CLE4EVR output is just dict {"image", "label"}
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader
