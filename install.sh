#!/bin/bash
# OpenPCSeg Installation Script (with Swin Transformer support)
#
# Usage:
#   bash install.sh
#
# This script automates the installation process described in docs/INSTALL.md

set -e  # Exit on error

echo "========================================================================"
echo "OpenPCSeg Installation Script"
echo "========================================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# echo -e "${GREEN}Step 1: Creating conda environment 'pcseg'${NC}"
# if conda env list | grep -q "^pcseg "; then
#     echo -e "${YELLOW}Environment 'pcseg' already exists. Skipping creation.${NC}"
# else
#     conda create -n pcseg python=3.9 -y
#     echo -e "${GREEN}✓ Environment created${NC}"
# fi

echo ""
echo -e "${GREEN}Step 2: Activating environment${NC}"
eval "$(conda shell.bash hook)"
conda activate fuse
echo -e "${GREEN}✓ Environment activated${NC}"

# echo ""
# echo -e "${GREEN}Step 3: Installing PyTorch with CUDA 11.3${NC}"
# conda install pytorch==2.0.0 torchvision cudatoolkit=11.8 -c pytorch -c conda-forge -y
# echo -e "${GREEN}✓ PyTorch installed${NC}"

# echo ""
# echo -e "${GREEN}Step 4: Installing PyTorch Scatter${NC}"
# conda install pytorch-scatter -c pyg -y
# echo -e "${GREEN}✓ PyTorch Scatter installed${NC}"

# echo ""
# echo -e "${GREEN}Step 5: Installing pip dependencies${NC}"
# pip install -r requirements.txt
# echo -e "${GREEN}✓ Pip dependencies installed${NC}"

echo ""
echo -e "${GREEN}Step 6: Building TorchSparse${NC}"
if [ ! -d "package/torchsparse_dir" ]; then
    echo "Creating torchsparse_dir..."
    mkdir -p package/torchsparse_dir
fi

cd package/

# Check if sparsehash is already compiled
if [ ! -d "torchsparse_dir/sphash" ]; then
    echo "Setting up sparsehash..."

    if [ -f "sparsehash.zip" ]; then
        unzip -q sparsehash.zip
        mv sparsehash-master/ sparsehash/
    fi

    if [ -d "sparsehash" ]; then
        cd sparsehash/
        ROOT_PATH=$(dirname $(dirname $(pwd)))
        ./configure --prefix=${ROOT_PATH}/package/torchsparse_dir/sphash/
        make
        make install
        cd ..
        echo -e "${GREEN}✓ Sparsehash built${NC}"
    else
        echo -e "${YELLOW}Warning: sparsehash.zip not found. Skipping.${NC}"
        echo "Please manually install sparsehash if needed."
    fi
else
    echo -e "${YELLOW}Sparsehash already built. Skipping.${NC}"
fi

# Install torchsparse
if [ -f "torchsparse.zip" ]; then
    if [ ! -d "torchsparse" ]; then
        unzip -q torchsparse.zip
    fi

    echo "Building torchsparse (this may take a while)..."
    pip install ./torchsparse
    echo -e "${GREEN}✓ TorchSparse installed${NC}"
else
    echo -e "${YELLOW}Warning: torchsparse.zip not found.${NC}"
    echo "Please install torchsparse manually from package/ directory."
fi

cd ..

echo ""
echo -e "${GREEN}Step 7: Building Range Image Library${NC}"
cd package/
if [ -f "range_lib.zip" ]; then
    if [ ! -d "range_lib" ]; then
        unzip -q range_lib.zip
    fi

    cd range_lib/
    python setup.py install
    cd ../..
    echo -e "${GREEN}✓ Range library installed${NC}"
else
    cd ..
    echo -e "${YELLOW}Warning: range_lib.zip not found.${NC}"
    echo "Please install range_lib manually from package/ directory."
fi

echo ""
echo -e "${GREEN}Step 8: Registering PCSeg package${NC}"
python setup.py develop
echo -e "${GREEN}✓ PCSeg registered${NC}"

echo ""
echo "========================================================================"
echo -e "${GREEN}Installation Complete!${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "     conda activate pcseg"
echo ""
echo "  2. Test the installation:"
echo "     python test_swin_range_branch.py"
echo ""
echo "  3. Start training:"
echo "     python train.py --cfg_file tools/cfgs/fusion/semantic_kitti/rpvnet_swin_tiny_cr17_5.yaml"
echo ""
echo "========================================================================"
