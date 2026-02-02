#!/bin/bash
# Build script for QSMbly WebAssembly components
# This compiles the Rust code to WASM and copies it to the serve directory

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$SCRIPT_DIR/rust-wasm"
WASM_DIR="$SCRIPT_DIR/wasm"

echo "=== QSMbly WASM Build ==="
echo ""

# Check for required tools
if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack is not installed."
    echo "Install it with: cargo install wasm-pack"
    echo "Or visit: https://rustwasm.github.io/wasm-pack/installer/"
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    echo "Error: cargo (Rust) is not installed."
    echo "Install from: https://rustup.rs/"
    exit 1
fi

# Build WASM
echo "[1/3] Building WASM with wasm-pack..."
cd "$RUST_DIR"
wasm-pack build --target web --release

echo ""
echo "[2/3] Copying WASM files to serve directory..."
cp "$RUST_DIR/pkg/qsm_wasm.js" "$WASM_DIR/"
cp "$RUST_DIR/pkg/qsm_wasm_bg.wasm" "$WASM_DIR/"
cp "$RUST_DIR/pkg/qsm_wasm.d.ts" "$WASM_DIR/" 2>/dev/null || true
cp "$RUST_DIR/pkg/qsm_wasm_bg.wasm.d.ts" "$WASM_DIR/" 2>/dev/null || true

# Copy romeo files if they exist
if [ -f "$RUST_DIR/pkg/romeo_wasm.js" ]; then
    cp "$RUST_DIR/pkg/romeo_wasm.js" "$WASM_DIR/"
    cp "$RUST_DIR/pkg/romeo_wasm_bg.wasm" "$WASM_DIR/"
    cp "$RUST_DIR/pkg/romeo_wasm.d.ts" "$WASM_DIR/" 2>/dev/null || true
    cp "$RUST_DIR/pkg/romeo_wasm_bg.wasm.d.ts" "$WASM_DIR/" 2>/dev/null || true
fi

echo ""
echo "[3/3] Build complete!"
echo ""
echo "WASM files in $WASM_DIR:"
ls -lh "$WASM_DIR"/*.wasm "$WASM_DIR"/*.js 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "To start the development server:"
echo "  python -m http.server 8080"
echo "  # Then open http://localhost:8080"
