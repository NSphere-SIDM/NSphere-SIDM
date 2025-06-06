# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2025-06-05

This release transforms NSphere from a purely gravitational spherical N-body simulator to implement Self-Interacting Dark Matter (SIDM) physics, dual density profile support, enhanced visualization capabilities, and updated documentation.

### Added

*   **SIDM Physics:**
    *   Self-Interacting Dark Matter simulation with configurable cross-section via `--sidm-kappa` (default: 50.0 cm²/g)
    *   Serial and parallel execution modes via `--sidm-mode` with automatic OpenMP fallback
    *   Monte Carlo scattering in center-of-mass frame
    *   Real-time scatter statistics tracking
    *   Integration with all gravitational integrators

*   **Dual Density Profiles:**
    *   NFW profile with power-law cutoff (default)
    *   Cored Plummer-like profile option via `--profile <type>`
    *   Profile-specific parameters: `--scale-radius`, `--halo-mass`, `--cutoff-factor`
    *   NFW-specific `--falloff-factor` for concentration parameter
    *   High-resolution splines (100,000 points) for accurate calculations

*   **Reproducibility Features:**
    *   Hierarchical seed management: `--master-seed`, `--init-cond-seed`, `--sidm-seed`
    *   Persistent seed files with automatic saving
    *   `--load-seeds` option to resume previous simulations
    *   Independent random number streams for each component

*   **Visualization Tools:**
    *   Tangential velocity histograms for angular momentum analysis
    *   High-resolution phase space plots (400×400 bins)
    *   Dynamic plot ranging with percentile-based scaling
    *   Core region animations in Jupyter notebook Example 2
    *   Log-scale for some density profiles and convergence plots

*   **Documentation System:**
    *   Sphinx documentation with LaTeX math support
    *   Integrated dynamic Jupyter notebook examples
    *   API documentation for C and Python components
    *   Reproducibility guidance with seed management examples

*   **Performance Features:**
    *   Persistent sort buffer optimization (3-6x speedup on Apple Silicon)
    *   Cross-platform disk space monitoring with warnings
    *   Dynamic cache detection for optimized sorting

### Changed

*   **Command-Line Interface:**
    *   Total options increased to 28 (12 new, 1 renamed)
    *   `--enable-log` renamed to `--log`
    *   Help text reorganized into 6 logical sections
    *   Unicode M☉ symbol for solar mass units
    *   Default SIDM mode changed from serial to parallel
    *   Default save mode changed to `all`

*   **Numerical Algorithms:**
    *   Adams-Bashforth integrator upgraded to reflect 3rd order
    *   Scattering-aware integration with automatic order reduction
    *   Enhanced bootstrap phase with 20 mini-substeps
    *   Profile-aware dynamical time calculations
    *   Improved GSL integration for better singularity handling

*   **Build System:**
    *   Architecture-specific optimizations for Apple Silicon
    *   Improved OpenMP detection and fallback
    *   FFTW3 path fixes for Apple Silicon Homebrew installations

*   **Data Visualization:**
    *   2D histograms upgraded to 400×400 resolution
    *   Velocity distributions normalized to probability densities
    *   Velocity distributionat fixed radius plot changed to 2× scale radius (profile-specific)
    *   Physical time units (Gyr) in animations
    *   Power-law color scaling for phase space visibility

### Fixed

*   **Physics Corrections:**
    *   4π factor restored in theoretical NFW gravitational potential
    *   double-scaling error removed from NFW mass calculations
    *   Integration variable transformations corrected

*   **Build Issues:**
    *   FFTW3 header paths fixed for Apple Silicon
    *   OpenMP detection improved with proper warnings
    *   Binary file mode issues resolved on Windows

*   **Reproducibility:**
    *   Seed file symlinks now use relative paths
    *   Cross-platform seed file handling improved

*   **Visualization:**
    *   2D histogram percentile calculation corrected
    *   Animation frame range includes final snapshot
    *   Matplotlib static image display bug resolved

*   **Documentation:**
    *   37 Doxygen rendering issues fixed
    *   MathJax configuration corrected for LaTeX
    *   Broken navigation links repaired

### Removed

*   **Legacy Code:**
    *   Some verbose debug print statements
    *   Unused global variables
    *   Orphaned documentation directory
    *   Redundant AB8-like integrator logic

### Security

*   Replaced 25+ instances of `sprintf` with `snprintf` to prevent buffer overflows

### Performance

*   Persistent sort buffer yields 3-6x runtime improvement on Apple Silicon
*   Parallel sorting optimizations provide 30-50% improvement in sort phase
*   O(N log N) scaling verified for large simulations

### Notes

*   This release fundamentally expands NSphere into a research tool for SIDM
*   All SIDM functionality is new 
*   NFW is the default profile for new simulations
*   Backward compatibility maintained for purely gravitational N-body simulations

---

## [0.1.2] - 2025-04-27

This release improves multiprocessing stability, enhances macOS plotting compatibility, and updates dependency handling with flexible version requirements. These updates, excluding the Virtual Environment update, were meant to be included in v0.1.1 but were inadvertently omitted from that package.

### Fixed

*   **Multiprocessing:** Eliminated repetitive "Could not verify project root" warnings by limiting messages to the main process; improved multiprocessing stability and configuration on macOS.
*   **macOS:** Addressed plotting backend issues for Dock icon behavior by setting Matplotlib to non-interactive 'Agg' backend globally.

### Changed

*   **Dependencies:** Updated `requirements.txt` to use flexible minimum version constraints (e.g., `>=X.Y`) instead of exact matches (`==X.Y.Z`).
*   **Build Process:** Improved project root path detection logic for robustness across execution contexts.

### Added

*   **Virtual Environment:** Enhanced activation script with clearer guidance when users forget to use the 'source' command.

## [0.1.1] - 2025-04-24

This release focuses on improving build system robustness across platforms (especially Windows and macOS), enhancing documentation, adding citation support, and fixing several bugs identified since the initial beta.

### Notes

*   **Correction:** While improvements to multiprocessing stability, macOS plotting compatibility, and dependency versioning were correctly described in the commit message and CHANGELOG.md for v0.1.1, the actual code changes for these features were inadvertently omitted when packaging the release. These changes are fully implemented in v0.1.2.

### Added

*   **Citation Support:** Implemented formal citation infrastructure (`CITATION.cff`) and validation workflow to ensure proper academic citations.
*   **Dynamic Cache Sizing:** Added dynamic L3 cache detection for optimized sorting performance on Linux and macOS.

### Fixed

*   **Windows Stability:** Resolved "Missing Rankfile Bug" by implementing GSL RNG instead of platform-specific `drand48()`; fixed binary file mode issues (`wb`/`rb`) in `nsphere.c`.
*   **Build System:** Corrected `QUAD_CACHE` macro redefinition warnings using GCC diagnostic pragmas.
*   **Build System (macOS):** Fixed `python`/`python3` command usage in Makefile; implemented robust header-based OpenMP detection.
*   **Build System (Windows):** Resolved path handling issues and removed `cmd.exe` shell coupling for MinGW/Clang; fixed `Makefile.Windows` comments preventing debug messages.

### Changed

*   **Windows Build:** Standardized recommended MSYS2 environment to CLANG64 (from MinGW64).
*   **Build Process:** Enhanced macOS build flow with integrated user prompts and error handling.
*   **Build Process:** Enhanced toolchain configuration with bidirectional sync between `USE_MSVC` and `USE_MINGW` flags.

### Documentation

*   **README Overhaul:** Added formal citation section with BibTeX entry, arXiv badge, quick navigation links, and improved platform-specific build/usage instructions (Windows, WSL2, PowerShell, MSVC).
*   **API Docs:** Clarified that Python API documentation serves both users and developers.
*   **Build Docs:** Added Pandoc installation instructions for Windows documentation builds.
*   **Build Docs:** Enhanced MSVC build instructions with comprehensive step-by-step guidance.

---

## [0.1.0] - 2025-04-22

*   Initial Beta Release.