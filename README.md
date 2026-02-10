# QSMbly: Browser-Based Quantitative Susceptibility Mapping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Ready-green.svg)](https://pages.github.com/)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-Powered-blue.svg)](https://webassembly.org/)

A complete **Quantitative Susceptibility Mapping (QSM)** pipeline that runs entirely in your web browser using WebAssembly. No installation, no backend servers, no data uploads - just pure client-side MRI processing.

[ACCESS QSMbly HERE](https://astewartau.github.io/qsmbly/)

## Features

- **Completely Private**: All processing happens locally in your browser - your data never leaves your computer
- **Zero Installation**: No Python, MATLAB, or specialized software required
- **Cross-Platform**: Works on Windows, macOS, Linux, and even mobile devices
- **Interactive**: Real-time visualization with NiiVue, adjustable contrast, and masking thresholds
- **Portable**: Static files can be hosted anywhere (GitHub Pages, local server, etc.)
- **Comprehensive**: 20+ algorithms covering the complete QSM pipeline

## Implemented Algorithms

### Phase Unwrapping (2 methods)

| Algorithm | Description |
|-----------|-------------|
| **ROMEO** | Region-growing with quality-guided ordering. Uses magnitude and gradient coherence weighting for robust unwrapping. |
| **Laplacian** | FFT-based Poisson solver approach. Fast and effective for well-conditioned data. |

### Background Field Removal (6 methods)

| Algorithm | Description |
|-----------|-------------|
| **SMV** | Spherical Mean Value - simple baseline subtraction method |
| **SHARP** | Sophisticated Harmonic Artifact Reduction for Phase data |
| **V-SHARP** | Variable-radius SHARP with multi-scale kernel approach |
| **PDF** | Projection onto Dipole Fields |
| **iSMV** | Iterative Spherical Mean Value deconvolution |
| **LBV** | Laplacian Boundary Value method |

### Dipole Inversion (9 methods)

| Algorithm | Description |
|-----------|-------------|
| **TKD** | Truncated K-space Division - fast closed-form solution |
| **TSVD** | Truncated Singular Value Decomposition |
| **Tikhonov** | L2 regularization with configurable kernels (identity, gradient, Laplacian) |
| **TV-ADMM** | Total Variation via ADMM - edge-preserving regularization |
| **NLTV** | Nonlinear Total Variation with iterative reweighting |
| **RTS** | Rapid Two-Step method (LSMR + TV refinement) |
| **MEDI** | Morphology-Enabled Dipole Inversion with gradient and SNR weighting |
| **iLSQR** | Iterative LSQR with streaking artifact removal (Li et al., 2015) |
| **TGV** | Total Generalized Variation - direct QSM from wrapped phase without separate unwrapping |

### Multi-Echo Processing

| Algorithm | Description |
|-----------|-------------|
| **MCPC-3DS** | Multi-Channel Phase Combination with 3D smoothing. Removes phase offsets across echoes. |
| **Weighted B0** | Field map calculation with multiple weighting strategies (SNR, variance, magnitude, TEs) |

### Advanced Reconstruction Methods

| Algorithm | Description |
|-----------|-------------|
| **QSMART** | Two-stage QSM with Artifact Reduction using Tissue maps. Uses Spatially Dependent Filtering (SDF) and Frangi vesselness to separate tissue and vasculature, reducing streaking artifacts from veins. |
| **TGV Single-Step** | Combines background removal and dipole inversion in one optimization. Useful for challenging data. |

### Additional Tools

- **Phase Scaling**: Automatic normalization to [-π, π] range
- **Brain Extraction (BET)**: Region-growing based brain masking with mesh evolution
- **Gaussian Smoothing**: 3D phase-aware smoothing with wrap handling

## Quick Start

### Option 1: GitHub Pages
1. Visit the live demo (if deployed)
2. Upload your magnitude and phase NIfTI files
3. Set acquisition parameters (Echo Time, Field Strength)
4. Run the pipeline

### Option 2: Local Development
```bash
# Clone the repository
git clone https://github.com/astewartau/qsmbly.git
cd qsmbly

# Serve locally (Python)
python -m http.server 8080

# Or use Node.js
npx serve .

# Open http://localhost:8080
```

## Input Requirements

### Required Files
- **Magnitude Image**: `.nii` or `.nii.gz` format
- **Phase Image**: `.nii` or `.nii.gz` format (same dimensions as magnitude)

### Acquisition Parameters
- **Echo Time (TE)**: In seconds (e.g., 0.004 for 4ms)
- **Magnetic Field Strength**: In Tesla (e.g., 3.0, 7.0)

### Recommended Acquisition
- **Sequence**: 3D Gradient Echo (GRE) or FLASH
- **Resolution**: ≤ 1mm³ isotropic
- **Field Strength**: 3T or 7T
- **Echo Time**: First echo or multi-echo sequence

## Repository Structure

```
qsmbly/
├── index.html              # Main application interface
├── build.sh                # WASM build script
├── package.json            # Node.js dependencies (testing)
├── jest.config.js          # Jest testing configuration
├── js/
│   ├── qsm-app-romeo.js    # Main application logic
│   ├── qsm-worker-pure.js  # Web worker for processing
│   ├── app/
│   │   └── config.js       # Centralized configuration
│   ├── modules/
│   │   ├── file-io/        # NIfTI file I/O utilities
│   │   │   └── NiftiUtils.js
│   │   ├── mask/           # Mask creation & morphology
│   │   │   ├── MorphologyOps.js  # Erosion, dilation, hole filling
│   │   │   └── ThresholdUtils.js # Otsu thresholding
│   │   ├── ui/             # UI utilities
│   │   │   ├── ConsoleOutput.js  # Real-time logging
│   │   │   └── ProgressManager.js
│   │   └── viewer/         # Visualization utilities
│   │       └── EchoNavigator.js  # Multi-echo navigation
│   ├── worker/
│   │   └── utils/          # Worker utility functions
│   │       ├── PhaseUtils.js     # Phase scaling, B0 calculation
│   │       ├── MaskUtils.js      # Threshold-based masking
│   │       └── FilterUtils.js    # 3D box filtering
│   └── test/
│       └── setup.js        # Jest test setup
├── css/
│   └── modern-styles.css   # Modern UI styling
├── wasm/                   # Compiled WebAssembly (served)
│   ├── qsm_wasm.js         # JS bindings
│   ├── qsm_wasm_bg.wasm    # WASM binary (~1 MB)
│   ├── romeo_wasm.js       # ROMEO-specific bindings
│   └── romeo_wasm_bg.wasm  # ROMEO binary (~22 KB)
├── rust-wasm/              # Rust source code
│   ├── Cargo.toml          # Rust dependencies
│   └── src/
│       ├── lib.rs          # WASM entry points (59 exports)
│       ├── fft.rs          # FFT with cached plans
│       ├── nifti_io.rs     # NIfTI file handling
│       ├── inversion/      # Dipole inversion algorithms
│       │   ├── tkd.rs      # TKD/TSVD
│       │   ├── tikhonov.rs # L2 regularization
│       │   ├── tv.rs       # TV-ADMM
│       │   ├── nltv.rs     # Nonlinear TV
│       │   ├── rts.rs      # RTS two-step
│       │   ├── medi.rs     # MEDI L1 optimization
│       │   ├── tgv.rs      # TGV from wrapped phase
│       │   └── ilsqr.rs    # iLSQR streaking removal
│       ├── bgremove/       # Background removal
│       │   ├── smv.rs, sharp.rs, vsharp.rs
│       │   ├── pdf.rs, ismv.rs, lbv.rs
│       │   └── sdf.rs      # Spatially Dependent Filtering
│       ├── unwrap/         # Phase unwrapping
│       │   ├── romeo.rs    # ROMEO algorithm
│       │   └── laplacian.rs
│       ├── kernels/        # Dipole, SMV, Laplacian kernels
│       ├── solvers/        # CG, LSMR solvers
│       ├── utils/          # Gradient ops, multi-echo, padding
│       │   ├── qsmart.rs   # QSMART two-stage reconstruction
│       │   ├── frangi.rs   # Frangi vesselness filter
│       │   ├── vasculature.rs    # Vessel detection
│       │   ├── multi_echo.rs     # Multi-echo processing
│       │   ├── curvature.rs      # Curvature calculations
│       │   ├── bias_correction.rs # Bias field correction
│       │   └── simd_ops.rs # SIMD-accelerated operations (optional)
│       └── bet/            # Brain extraction
├── python/                 # Reference implementations
└── other/                  # Julia reference code
```

## Building from Source

### Prerequisites
1. **Install Rust**: https://rustup.rs/
2. **Install wasm-pack**:
   ```bash
   cargo install wasm-pack
   ```

### Build Process
```bash
# Standard build (maximum browser compatibility)
./build.sh

# SIMD-accelerated build (faster, but requires modern browsers)
./build.sh --simd

# Or manually:
cd rust-wasm
wasm-pack build --target web --release              # Standard
wasm-pack build --target web --release --features simd  # With SIMD
cp pkg/qsm_wasm.js ../wasm/
cp pkg/qsm_wasm_bg.wasm ../wasm/
```

### SIMD Acceleration

The `--simd` flag enables 128-bit SIMD vectorization for faster processing of iterative algorithms (MEDI, TV, TGV, etc.). This provides approximately **2-4x speedup** for element-wise operations.

**Browser Requirements for SIMD:**
| Browser | Minimum Version |
|---------|-----------------|
| Chrome  | 91+ (May 2021)  |
| Firefox | 89+ (June 2021) |
| Safari  | 16.4+ (March 2023) |
| Edge    | 91+ (May 2021)  |

If targeting older browsers, use the standard build without `--simd`.

### Running Locally
```bash
python -m http.server 8080
# Open http://localhost:8080
```

Note: You may need to hard-refresh (Ctrl+Shift+R) to clear cached WASM files after rebuilding.

### Running Tests
```bash
# Install dependencies
npm install

# Run all tests
npm test

# Watch mode for development
npm run test:watch

# Generate coverage report
npm run test:coverage
```

## Technical Stack

### Rust/WebAssembly
- **wasm-bindgen**: JavaScript/WASM interop
- **rustfft**: FFT computations
- **nifti**: NIfTI file handling
- **ndarray**: N-dimensional arrays
- **wide**: Optional SIMD acceleration (128-bit vectorization)

### Frontend
- **NiiVue**: WebGL neuroimaging viewer
- **Pyodide**: Python in browser (optional)

## References

1. **ROMEO**: Dymerska et al. "Phase unwrapping with a rapid opensource minimum spanning tree algorithm (ROMEO)." *Magnetic Resonance in Medicine* (2021)
2. **Laplacian Unwrapping**: Schofield & Zhu. "Fast phase unwrapping algorithm for interferometric applications." *Optics Letters* (2003)
3. **SHARP**: Schweser et al. "Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase." *MRM* (2011)
4. **V-SHARP**: Li et al. "A method for estimating and removing streaking artifacts in quantitative susceptibility mapping." *NeuroImage* (2015)
5. **PDF**: Liu et al. "A novel background field removal method for MRI using projection onto dipole fields." *NMR in Biomedicine* (2011)
6. **LBV**: Zhou et al. "Background field removal by solving the Laplacian boundary value problem." *NMR in Biomedicine* (2014)
7. **TKD**: Shmueli et al. "Magnetic susceptibility mapping of brain tissue in vivo using MRI phase data." *MRM* (2009)
8. **MEDI**: Liu et al. "Morphology enabled dipole inversion (MEDI) from a single-angle acquisition." *MRM* (2011)
9. **TGV-QSM**: Langkammer et al. "Fast quantitative susceptibility mapping using 3D EPI and total generalized variation." *NeuroImage* (2015)
10. **RTS**: Kames et al. "Rapid two-step dipole inversion for susceptibility mapping with sparsity priors." *NeuroImage* (2018)
11. **MCPC-3DS**: Eckstein et al. "Computationally efficient combination of multi-channel phase data from multi-echo acquisitions (ASPIRE)." *MRM* (2018)
12. **QSMART**: Özbay et al. "A comprehensive numerical analysis of background phase correction with V-SHARP." *NMR in Biomedicine* (2017)
13. **iLSQR**: Li et al. "A method for estimating and removing streaking artifacts in quantitative susceptibility mapping." *NeuroImage* (2015)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NiiVue Team**: For the excellent neuroimaging viewer
- **QSM Community**: For developing and sharing algorithms
- **ROMEO/MriResearchTools.jl**: Reference implementations

## Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/astewartau/qsmbly/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/astewartau/qsmbly/discussions)
