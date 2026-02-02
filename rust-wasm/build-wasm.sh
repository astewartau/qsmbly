#!/bin/bash
set -e

echo "Building QSM WASM module..."

# Navigate to rust-wasm directory
cd "$(dirname "$0")"

# Ensure rustup toolchain is available
export PATH="$HOME/.cargo/bin:$PATH"

# Add wasm target if not already present
rustup target add wasm32-unknown-unknown 2>/dev/null || true

# Build with wasm-pack
wasm-pack build --target web --release

# Create wasm output directory if it doesn't exist
mkdir -p ../wasm

# Copy generated files to wasm directory
cp pkg/qsm_wasm.js ../wasm/
cp pkg/qsm_wasm_bg.wasm ../wasm/

echo "Build complete! Files copied to ../wasm/"
echo "  - qsm_wasm.js"
echo "  - qsm_wasm_bg.wasm"

# Show file sizes
ls -lh ../wasm/qsm_wasm*
