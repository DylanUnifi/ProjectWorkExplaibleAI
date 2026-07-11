# data_loader/utils.py
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch
from torchvision.datasets import SVHN
import numpy as np
from sklearn.decomposition import PCA


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


def load_dataset_by_name(name, binary_classes=None, grayscale=True, root='./data'):
    """
    Load (train_dataset, test_dataset) filtered on `binary_classes` and binary labels {0,1}.
    - fashion_mnist : Subset of torchvision.FashionMNIST
    - cifar10       : Subset of torchvision.CIFAR10
    - svhn          : TensorDataset (X, y) with y in int64 (torch.long)
    """
    if binary_classes is None:
        binary_classes = [3, 8]

    name = str(name).lower()

    if name == 'fashion_mnist':
        transform = build_transform(grayscale=True)
        train_set = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_set  = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

        train_idx = [i for i, t in enumerate(train_set.targets) if int(t) in binary_classes]
        test_idx  = [i for i, t in enumerate(test_set.targets)  if int(t) in binary_classes]

        train_subset = relabel_subset(Subset(train_set, train_idx), train_set.targets, binary_classes)
        test_subset  = relabel_subset(Subset(test_set,  test_idx),  test_set.targets,  binary_classes)
        return train_subset, test_subset

    elif name == 'cifar10':
        transform = build_transform(grayscale=grayscale, augment=True)
        train_set = datasets.CIFAR10(root=root, train=True,  download=True, transform=transform)
        test_set  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

        train_idx = [i for i, t in enumerate(train_set.targets) if int(t) in binary_classes]
        test_idx  = [i for i, t in enumerate(test_set.targets)  if int(t) in binary_classes]

        train_subset = relabel_subset(Subset(train_set, train_idx), train_set.targets, binary_classes)
        test_subset  = relabel_subset(Subset(test_set,  test_idx),  test_set.targets,  binary_classes)
        return train_subset, test_subset

    elif name == 'svhn':
        # SVHN returns labels in {0..9}, here we filter, and return y in torch.long
        transform = build_transform(grayscale=grayscale, augment=False)
        train_set = SVHN(root=root, split='train', download=True, transform=transform)
        test_set  = SVHN(root=root, split='test',  download=True, transform=transform)

        def filter_and_process(dataset):
            X, y = [], []
            for idx in range(len(dataset)):
                img, label = dataset[idx]
                label = int(label)
                if label in binary_classes:
                    X.append(img)
                    y.append(1 if label == int(binary_classes[1]) else 0)
            X = torch.stack(X) if len(X) else torch.empty(0)
            y = torch.tensor(y, dtype=torch.long)  # int64 for sklearn
            return torch.utils.data.TensorDataset(X, y)

        train_dataset = filter_and_process(train_set)
        test_dataset  = filter_and_process(test_set)
        return train_dataset, test_dataset

    else:
        raise ValueError(f"Dataset inconnu: {name}")


def extract_features_pca(dataloader, n_components, pca_model=None, desc="Extracting features"):
    """
    Extract flattened image features from a dataloader and apply PCA.

    Uses IncrementalPCA when fitting so that the full n_samples×n_features matrix
    (e.g. 9000×150528 ≈ 5 GB) is never materialised in RAM.  Two tqdm passes are
    shown: one for fitting, one for the final transform.

    Supports both:
    - dict batches with keys "image" and "label" (CLEVR-Hans loaders)
    - tuple/list batches of (images, labels)
    """
    from tqdm import tqdm
    from sklearn.decomposition import IncrementalPCA

    def _unpack(batch):
        if isinstance(batch, dict):
            return batch["image"], batch["label"]
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("Unsupported batch format in extract_features_pca")

    needs_fit = (pca_model is None)

    # ── Pass 1: incremental PCA fit (train only) ─────────────────
    if needs_fit:
        pca_model = IncrementalPCA(n_components=n_components)
        # Accumulate small batches so partial_fit always sees >= n_components samples
        min_chunk = max(n_components * 4, 128)
        buffer = []
        buf_size = 0
        print(f"   Fitting IncrementalPCA ({n_components} components, chunk≥{min_chunk})...", flush=True)
        for batch in tqdm(dataloader, desc=f"{desc} [fit]", unit="batch"):
            images, _ = _unpack(batch)
            bf = images.view(images.size(0), -1).detach().cpu().numpy().astype(np.float32)
            buffer.append(bf)
            buf_size += len(bf)
            if buf_size >= min_chunk:
                pca_model.partial_fit(np.concatenate(buffer, axis=0))
                buffer, buf_size = [], 0
        if buf_size >= n_components:
            pca_model.partial_fit(np.concatenate(buffer, axis=0))
        var = pca_model.explained_variance_ratio_.sum() * 100
        print(f"   PCA fitted — variance explained: {var:.1f}%", flush=True)

    # ── Pass 2: transform (all splits) ───────────────────────────
    features, labels = [], []
    transform_desc = f"{desc} [transform]" if needs_fit else desc
    for batch in tqdm(dataloader, desc=transform_desc, unit="batch"):
        images, batch_labels = _unpack(batch)
        bf = images.view(images.size(0), -1).detach().cpu().numpy().astype(np.float32)
        features.append(pca_model.transform(bf))
        labels.append(batch_labels.detach().cpu().numpy())

    X_reduced = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    return X_reduced.astype(np.float32, copy=False), y, pca_model


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
        if self.return_confounders and self.confounders:
            class_key = label
            if class_key in self.confounders:
                is_confounded = self.confounders[class_key]["confounded"]
                confounder_info = {
                    "is_confounded": is_confounded,
                    "confounder_attrs": self.confounders[class_key].get("attributes", []),
                }

        # Group index for GroupDRO: 0=confounded, 1=clean
        group_idx = 0 if is_confounded else 1

        return {
            "image": image,
            "label": label,
            "confounder_info": confounder_info,
            "image_id": img_filename,
            "group_idx": group_idx,
        }


def clevr_collate_fn(batch):
    """Custom collate that handles variable-length confounder_info."""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    image_ids = [item["image_id"] for item in batch]
    confounder_infos = [item["confounder_info"] for item in batch]
    group_idx = torch.tensor([item["group_idx"] for item in batch], dtype=torch.long)

    return {
        "image": images,
        "label": labels,
        "confounder_info": confounder_infos,
        "image_id": image_ids,
        "group_idx": group_idx,
    }


def get_clevr_hans_loaders(
    root_dir,
    variant="clevr_hans3",
    batch_size=64,
    num_workers=8,
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
                # MNMath RSBench returns the sum of digits.
                # We extract the sum directly to enable multi-class prediction (0-18).
                primary_label = int(label_data[0])
            else:
                primary_label = int(label_data)
                
            self.labels.append(primary_label)
            
            concept_values = data["meta"]["concepts"]
            concepts = np.array(concept_values).astype(np.int64)
            self.concepts.append(concepts)
            
        self.list_images = new_images
        
        print(f"MNMath {split}: {len(self.list_images)} images loaded.")
        
    def _extract_number(self, path):
        match = re.search(r"\d+", path)
        return int(match.group()) if match else 0

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        img_path = self.list_images[idx]
        image = Image.open(img_path).convert("L")  # Grayscale
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        concepts = self.concepts[idx]
        
        # Format identical to CLEVR-Hans for compatibility with train_all_models.py
        return {
            "image": image,
            "label": label,
            "group": 0,  # default
            "scene": {"concepts": concepts.tolist()},
            "confounder_info": {},
            "image_path": str(img_path)
        }


def get_mnmath_loaders(
    root_dir="./data/mnmath",
    batch_size=32,
    num_workers=4,
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
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
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
        return train_loader, val_loader, test_loader
    except Exception as e:
        print(f"Warning: MNMath dataset not found or missing dependency: {e}. Returning empty loaders.")
        return None, None, None


