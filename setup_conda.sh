#!/bin/bash
# Setup script for Conda/Anaconda environment

set -e  # Exit on error

echo "=========================================="
echo "Setting up Conda environment"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: Conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓ Conda found: $(conda --version)${NC}"

# Check for CUDA
echo -e "\n${YELLOW}Checking for CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo -e "${GREEN}✓ CUDA detected: $CUDA_VERSION${NC}"
    HAS_CUDA=true
else
    echo -e "${YELLOW}⚠ CUDA not detected (will use CPU-only PyTorch)${NC}"
    HAS_CUDA=false
fi

# Ask about PyTorch installation
echo -e "\n${YELLOW}PyTorch Installation:${NC}"
if [ "$HAS_CUDA" = true ]; then
    echo "1) PyTorch with CUDA 11.8"
    echo "2) PyTorch with CUDA 12.1"
    echo "3) PyTorch CPU-only"
    read -p "Enter choice (1, 2, or 3): " pytorch_choice
else
    echo "Installing CPU-only PyTorch (no CUDA detected)"
    pytorch_choice="3"
fi

# Set environment file
ENV_FILE="environment.yml"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: $ENV_FILE not found${NC}"
    exit 1
fi

# Modify environment.yml based on PyTorch choice
echo -e "\n${YELLOW}Preparing environment file...${NC}"
cp environment.yml environment.yml.bak

# Comment out all PyTorch options first
sed -i 's/^  - pytorch::pytorch/  # - pytorch::pytorch/g' environment.yml
sed -i 's/^  - pytorch::cpuonly/  # - pytorch::cpuonly/g' environment.yml

# Uncomment the selected option
if [ "$pytorch_choice" = "1" ]; then
    echo "Configuring for CUDA 11.8..."
    sed -i '19s/^  # /  /' environment.yml  # Uncomment pytorch line
    sed -i '20s/^  # /  /' environment.yml  # Uncomment pytorch-cuda=11.8 line
elif [ "$pytorch_choice" = "2" ]; then
    echo "Configuring for CUDA 12.1..."
    sed -i '23s/^  # /  /' environment.yml  # Uncomment pytorch line
    sed -i '24s/^  # /  /' environment.yml  # Uncomment pytorch-cuda=12.1 line
else
    echo "Configuring for CPU-only..."
    sed -i '27s/^  # /  /' environment.yml  # Uncomment pytorch line
    sed -i '28s/^  # /  /' environment.yml  # Uncomment cpuonly line
fi

# Get environment name from file
ENV_NAME=$(grep "^name:" "$ENV_FILE" | awk '{print $2}')

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "\n${YELLOW}Environment '$ENV_NAME' already exists.${NC}"
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Updating environment...${NC}"
        conda env update -f "$ENV_FILE" --prune
        echo -e "${GREEN}✓ Environment updated${NC}"
    else
        echo "Using existing environment."
    fi
else
    echo -e "\n${YELLOW}Creating new environment '$ENV_NAME'...${NC}"
    conda env create -f "$ENV_FILE"
    echo -e "${GREEN}✓ Environment created${NC}"
fi

# Activate environment
echo -e "\n${YELLOW}Activating environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install package in editable mode
echo -e "\n${YELLOW}Installing package in editable mode...${NC}"
pip install -e .
echo -e "${GREEN}✓ Package installed${NC}"

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"
python -c "from joint_embedding_model import create_joint_embedding_model; print('✓ Import successful')"

# Check CUDA availability
echo -e "\n${YELLOW}Checking CUDA availability...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
    python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"
fi

# Summary
echo -e "\n=========================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "=========================================="
echo ""
echo "Environment name: $ENV_NAME"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
echo ""
echo "Quick start commands:"
echo "  make help          - Show all available commands"
echo "  make example       - Run example usage"
echo "  make train         - Start training"
echo "  make test          - Run tests"
echo ""
echo "To update the environment:"
echo "  conda env update -f $ENV_FILE --prune"
echo ""
echo "See DEVELOPMENT.md for detailed development guide."
echo ""
