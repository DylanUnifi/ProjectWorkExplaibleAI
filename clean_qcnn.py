import re
import os

def update_file(filename, replacements, regex_replacements):
    if not os.path.exists(filename): return
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    for pattern, new in regex_replacements:
        content = re.sub(pattern, new, content, flags=re.DOTALL)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

# train.py
update_file("train.py", [
    (", CLEVRQCNNClassifier", ""),
    ("choices=[\"resnet50\", \"vit\", \"protopnet\", \"hybrid_qcnn\"]", "choices=[\"resnet50\", \"vit\", \"protopnet\"]")
], [
    (r"    elif args\.model == \"hybrid_qcnn\":.*?n_layers=1\)", "")
])

# explain.py
update_file("explain.py", [
    (", CLEVRQCNNClassifier", ""),
    ("choices=[\"resnet50\", \"vit\", \"protopnet\", \"hybrid_qcnn\"]", "choices=[\"resnet50\", \"vit\", \"protopnet\"]")
], [
    (r"    elif args\.model == \"hybrid_qcnn\":.*?n_layers=1\)", "")
])

# docker-compose.yaml
update_file("docker-compose.yaml", [], [
    (r"  train_hybrid_qcnn_clevr:.*?(?=\n  #|\n$)", "")
])

# README.md
update_file("README.md", [
    ("  - `Hybrid QCNN` included as an experimental quantum-classical bonus.\n", ""),
    ("docker compose run --rm train_hybrid_qcnn_clevr\n", ""),
    (", HybridQCNN", ""),
    (" (ResNet-50, ViT, QCNN)", " (ResNet-50, ViT)")
], [])

print("Cleaned up scripts")
