#!/bin/bash
# ==========================================
# Script to download and prepare datasets
# ==========================================

echo "Creating dataset directories..."
mkdir -p data

echo "1) Downloading CLEVR-Hans3..."
if [ ! -d "CLEVR-Hans3" ]; then
    wget -nc -q --show-progress https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2611/CLEVR-Hans3.zip -P data/
    echo "Extracting CLEVR-Hans3..."
    unzip -q -n data/CLEVR-Hans3.zip -d .
    rm data/CLEVR-Hans3.zip
else
    echo "CLEVR-Hans3 already exists."
fi

echo "2) Setting up MNMath (MNISTMath)..."
if [ ! -d "data/mnmath" ]; then
    # MNMath is generated via rsbench-code
    if [ ! -d "rsbench-code" ]; then
        git clone https://github.com/unitn-sml/rsbench-code.git
    fi
    cd rsbench-code/rssgen
    # Remove bpy and mathutils (Blender Python libs) from requirements since they are not needed for MNMath and break on Python 3.10/runtime images
    sed -i '/bpy/d' requirements.txt
    sed -i '/mathutils/d' requirements.txt
    # Fix the broken 404 Yann LeCun MNIST URL by using the reliable AWS PyTorch mirror
    sed -i 's|http://yann.lecun.com/exdb/mnist/|https://ossci-datasets.s3.amazonaws.com/mnist/|g' rssgen/generators/mnist_generator.py
    pip install -r requirements.txt
    echo "Generating MNMath dataset... this may take a moment."
    python -m rssgen examples_config/mnist.yml mnist ../../data/mnmath
    cd ../..
    rm -rf rsbench-code
else
    echo "MNMath already exists in data/mnmath."
fi

echo "Done! Datasets are ready for the XAI pipelines."
