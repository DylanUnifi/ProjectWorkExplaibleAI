# data_loader/bdd_oia_loader.py

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path

class BDDOIADataset(Dataset):
    """
    BDD-OIA Dataset Loader.
    
    Structure attendue:
    bdd_oia/
    ├── videos/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── annotations/
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    """
    
    # 4 action categories
    ACTION_CLASSES = ["forward", "stop", "left_turn", "right_turn"]
    
    # 21 explanation categories
    EXPLANATION_CLASSES = [
        "red_light", "green_light", "pedestrian_crossing", "vehicle_ahead",
        "lane_change", "intersection", "road_sign", "traffic_cone",
        "bicycle", "motorcycle", "bus", "truck", "construction",
        "merge", "yield", "no_parking", "speed_limit", "crosswalk",
        "railroad", "school_zone", "fog"
    ]
    
    def __init__(
        self,
        root_dir,
        split="train",
        n_frames=16,           # Sample 16 frames from 5s video (150 frames @ 30fps)
        frame_size=(224, 224),
        transform=None,
        load_explanations=True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.transform = transform
        self.load_explanations = load_explanations
        
        # Load annotations
        anno_path = self.root_dir / "annotations" / f"{split}.json"
        with open(anno_path, "r") as f:
            self.annotations = json.load(f)
        
        self.video_paths = list(self.annotations.keys())
        
        print(f"📹 BDD-OIA {split}: {len(self.video_paths)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_id = self.video_paths[idx]
        anno = self.annotations[video_id]
        
        # Load video
        video_path = self.root_dir / "videos" / self.split / f"{video_id}.mp4"
        frames = self._load_video(video_path)  # (T, H, W, C)
        
        # Sample frames uniformly
        indices = np.linspace(0, len(frames) - 1, self.n_frames, dtype=int)
        frames = frames[indices]  # (n_frames, H, W, C)
        
        # Resize
        frames = np.array([
            cv2.resize(frame, self.frame_size) for frame in frames
        ])  # (T, H, W, C)
        
        # To tensor: (C, T, H, W)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        
        # Action label (0-3)
        action = self.ACTION_CLASSES.index(anno["action"])
        
        # Explanation labels (multi-label binary vector)
        if self.load_explanations:
            explanations = torch.zeros(len(self.EXPLANATION_CLASSES))
            for exp in anno.get("explanations", []):
                if exp in self.EXPLANATION_CLASSES:
                    explanations[self.EXPLANATION_CLASSES.index(exp)] = 1
        else:
            explanations = torch.zeros(len(self.EXPLANATION_CLASSES))
        
        if self.transform:
            frames = self.transform(frames)
        
        return {
            "video": frames,              # (C, T, H, W)
            "action": action,             # int (0-3)
            "explanations": explanations, # (21,) binary vector
            "video_id": video_id,
        }
    
    def _load_video(self, video_path):
        """Load all frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return np.array(frames)


def get_bdd_oia_loaders(root_dir, batch_size=16, num_workers=8, n_frames=16):
    """Get train/val/test loaders."""
    
    train_dataset = BDDOIADataset(root_dir, split="train", n_frames=n_frames)
    val_dataset = BDDOIADataset(root_dir, split="val", n_frames=n_frames)
    test_dataset = BDDOIADataset(root_dir, split="test", n_frames=n_frames)
    
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
