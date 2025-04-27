# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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