# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- ERK4(3)-IP solver: Embedded Runge-Kutta 4(3) method in Interaction Picture with adaptive stepping
  - Bogacki-Shampine coefficients for error estimation
  - FSAL (First Same As Last) optimization
  - Error-based step size control with configurable tolerances
  - Typically uses fewer steps than fixed-step RK4IP while maintaining accuracy
- Modular solver architecture with `src/solvers/` directory
  - `ssfm.jl`: Split-Step Fourier Method implementation
  - `rk4ip.jl`: RK4 in Interaction Picture with OrdinaryDiffEq.jl
  - `erk4ip.jl`: Embedded RK4(3) with adaptive stepping
- Core types: `Medium`, `SimParams`, `Grid`, `Pulse`
- Three Raman response models: Blow-Wood, Lin-Agrawal, Hollenbeck
- Pulse generation functions: sech, Gaussian, CW, and custom
- Comprehensive utility functions for analysis
- Full documentation with examples
- Test suite with multiple test sets

### Changed
- Reorganized solver code into separate files in `src/solvers/` directory
- API update: `FiberParams` renamed to `Medium` for better clarity
- Updated benchmarks to test all three solvers (SSFM, RK4IP, ERK4IP)

### Fixed
- **Critical:** Gaussian pulse FWHM conversion formula corrected
  - Changed from `T₀ = FWHM / (2√(2ln2))` to `T₀ = FWHM / (2√ln2)`
  - Reduced FWHM error from 26.74% to 2.56% (within discretization tolerance)
  - Sech pulse formula was already correct

### Removed
- Deleted examples directory (moved to separate examples repository)
- Removed temporary work markdown files

### Features
- Higher-order dispersion (arbitrary order via Taylor expansion)
- Kerr nonlinearity (self-phase modulation)
- Self-steepening (shock term)
- Stimulated Raman scattering with multiple models
- Frequency-dependent loss support
- Adaptive step size control (RK4IP and ERK4IP)
- Energy conservation checks
- Type-stable implementations for high performance
- Three solver methods: `:ssfm`, `:rk4ip`, `:erk4ip`

## [0.1.0] - 2024-11-26

### Added
- Initial release of JuGNLSE.jl
- Basic GNLSE solver infrastructure
- RK4IP and SSFM integration methods
- Core physics implementations
- Documentation and examples
- Comprehensive test suite

[Unreleased]: https://github.com/brian-sinquin/JuGNLSE.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/brian-sinquin/JuGNLSE.jl/releases/tag/v0.1.0
