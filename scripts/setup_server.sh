#!/bin/bash
# Setup script for RTX 4090 training server
# Installs all dependencies for temporal + VLA (π0.5 and Groot)
#
# Usage:
#   git clone --recursive <repo-url>
#   cd temporal
#   bash scripts/setup_server.sh
#
# Prerequisites:
#   - Python 3.11+ (uv will manage this)
#   - CUDA 12.x (for GPU support)
#   - uv (Python package manager)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "================================================"
echo "  TempoRAL VLA Setup — RTX 4090 Server"
echo "================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# --- 1. Check prerequisites ---
echo "[1/7] Checking prerequisites..."

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "  uv: $(uv --version)"
echo "  Python target: >=3.11"

# --- 2. Check external repos ---
echo ""
echo "[2/7] Checking external dependencies (openpi, Isaac-GR00T)..."

if [ ! -d "$PROJECT_ROOT/openpi/src" ]; then
    echo "  ⚠ openpi not found. Cloning..."
    git clone https://github.com/Physical-Intelligence/openpi.git "$PROJECT_ROOT/openpi"
else
    echo "  ✓ openpi found"
fi

if [ ! -d "$PROJECT_ROOT/Isaac-GR00T/gr00t" ]; then
    echo "  ⚠ Isaac-GR00T not found. Cloning..."
    git clone https://github.com/NVIDIA/Isaac-GR00T.git "$PROJECT_ROOT/Isaac-GR00T"
else
    echo "  ✓ Isaac-GR00T found"
fi

# --- 3. Create venv and install base deps ---
echo ""
echo "[3/7] Setting up Python environment..."

uv sync
echo "  ✓ Base dependencies installed"

# --- 4. Install VLA dependencies ---
echo ""
echo "[4/7] Installing VLA dependencies..."

uv pip install -e ".[vla,groot]"
echo "  ✓ VLA + Groot dependencies installed"

# --- 5. Apply transformers_replace patches (π0.5) ---
echo ""
echo "[5/7] Applying transformers_replace patches for π0.5..."

OPENPI_REPLACE="$PROJECT_ROOT/openpi/src/openpi/models_pytorch/transformers_replace"
TRANSFORMERS_DIR="$PROJECT_ROOT/.venv/lib/python3.*/site-packages/transformers"

# Expand the glob
TRANSFORMERS_DIR=$(echo $TRANSFORMERS_DIR)

if [ -d "$OPENPI_REPLACE" ] && [ -d "$TRANSFORMERS_DIR" ]; then
    # Backup and apply patches
    for patch_file in "$OPENPI_REPLACE"/*.py; do
        filename=$(basename "$patch_file")
        # Find the target in transformers
        target=$(find "$TRANSFORMERS_DIR" -name "$filename" -type f | head -1)
        if [ -n "$target" ]; then
            if [ ! -f "${target}.bak" ]; then
                cp "$target" "${target}.bak"
            fi
            cp "$patch_file" "$target"
            echo "  Patched: $filename"
        fi
    done
    echo "  ✓ transformers_replace patches applied"
else
    echo "  ⚠ Skipping patches (openpi or transformers not found)"
    echo "    OPENPI_REPLACE=$OPENPI_REPLACE"
    echo "    TRANSFORMERS_DIR=$TRANSFORMERS_DIR"
fi

# --- 6. Verify imports ---
echo ""
echo "[6/7] Verifying imports..."

.venv/bin/python -c "
import sys
sys.path.insert(0, 'src')

# Test base package
from temporal.models.metacontroller import MetaController
print('  ✓ Base temporal package')

# Test VLA metacontroller
from temporal.vla.models.metacontroller_vla import VLAMetaController, VLAMetaControllerConfig
print('  ✓ VLA MetaController')

# Test OpenPI shims + PI0Pytorch
from temporal.vla.models._openpi_shims import install_shims
install_shims()
sys.path.insert(0, '$PROJECT_ROOT/openpi/src')
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
print('  ✓ PI0Pytorch (π0.5)')

# Test Groot
sys.path.insert(0, '$PROJECT_ROOT/Isaac-GR00T')
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
print('  ✓ Gr00tN1d6 (Groot)')

# Test wrappers
from temporal.vla.models.pi05_wrapper import Pi05Wrapper, Pi05WrapperConfig
from temporal.vla.models.groot_wrapper import GrootWrapper, GrootWrapperConfig
print('  ✓ VLA Wrappers')

# Check GPU
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  ✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)')
else:
    print('  ⚠ No GPU detected (CPU mode)')
"

# --- 7. Run tests ---
echo ""
echo "[7/7] Running tests..."

uv run pytest tests/ -v --tb=short 2>&1 | tail -20

echo ""
echo "================================================"
echo "  Setup complete!"
echo ""
echo "  Quick start:"
echo "    uv run python scripts/run_vla.py --model pi05 --dummy-data --quick"
echo "    uv run python scripts/run_vla.py --model groot --dummy-data --quick"
echo ""
echo "  With model weights:"
echo "    uv run python scripts/run_vla.py --model pi05 --config configs/pi05_metacontroller.yaml"
echo "    uv run python scripts/run_vla.py --model groot --config configs/groot_metacontroller.yaml"
echo "================================================"
