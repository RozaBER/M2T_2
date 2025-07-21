#!/bin/bash

echo "=== Setting up Weights & Biases (wandb) ==="

# Check if wandb is already configured
if [ -f ~/.netrc ] && grep -q "api.wandb.ai" ~/.netrc; then
    echo "✓ wandb is already configured"
else
    echo "wandb is not configured. You have two options:"
    echo ""
    echo "Option 1: Use wandb offline mode (no account needed)"
    echo "  export WANDB_MODE=offline"
    echo ""
    echo "Option 2: Login to wandb (requires account)"
    echo "  wandb login"
    echo ""
    echo "Setting to offline mode by default..."
    export WANDB_MODE=offline
    echo "export WANDB_MODE=offline" >> ~/.bashrc
fi

# Create wandb directory
mkdir -p ./wandb

echo ""
echo "wandb setup complete. Training will log to:"
echo "  - Online: https://wandb.ai/your-username/brain-to-text-optimized"
echo "  - Offline: ./wandb/latest-run/"