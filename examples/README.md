# Examples

This directory contains example scripts demonstrating the usage of JuGNLSE for various nonlinear fiber optics simulations.

## Philosophy

Examples serve different purposes than tests:
- **Tests**: Verify correctness, fast execution, minimal output
- **Examples**: Demonstrate features, educational, generate figures

## Available Examples

### Soliton Examples
- **`soliton_propagation.jl`** - Fundamental (N=1) soliton propagation
  - Demonstrates perfect balance between dispersion and nonlinearity
  - Shows shape preservation, energy conservation
  - Key concept: P₀ = |β₂|/(γ·T₀²)
  - Runtime: ~10 seconds
  
- **`higher_order_soliton.jl`** - N=2 soliton periodic compression
  - Demonstrates breathing behavior of higher-order solitons
  - Shows periodic return to initial shape
  - Includes compression ratio analysis
  - Runtime: ~15 seconds

## Running Examples

```julia
# From repository root
julia --project=. examples/soliton_propagation.jl

# Or from Julia REPL
julia> using Pkg; Pkg.activate(".")
julia> include("examples/soliton_propagation.jl")
```

## Important: T0 vs FWHM Convention

For soliton calculations, **always use `T0=true`** when creating pulses:

```julia
# CORRECT - explicitly uses T0 (1/e half-width)
pulse = sech_pulse(grid, T0, P0, λ0, T0=true)

# WRONG - treats T0 as FWHM, results in 3.1x insufficient power
pulse = sech_pulse(grid, T0, P0, λ0)
```

Soliton power depends on T₀: P₀ = |β₂|/(γ·T₀²). The relationship: FWHM ≈ 1.763·T₀.

## Output

Examples generate figures saved in the current directory:
- `soliton_propagation.png` - N=1 soliton results
- `higher_order_soliton.png` - N=2 soliton dynamics

## Requirements

Examples require Plots.jl for visualization:

```julia
using Pkg
Pkg.add("Plots")
```

## Planned Examples

### Basic Examples
- `01_linear_dispersion.jl` - Pulse broadening due to GVD
- `02_self_phase_modulation.jl` - SPM and spectral broadening

### Advanced Examples
- `05_supercontinuum_generation.jl` - Reproduce Dudley RMP 2006 Fig. 3
- `06_dispersive_wave.jl` - Phase-matched dispersive wave generation
- `07_raman_soliton_shift.jl` - Soliton self-frequency shift (SSFS)
- `08_shock_formation.jl` - Self-steepening and shock

### Feature Demonstrations
- `09_solver_comparison.jl` - Compare SSFM, RK4IP, ERK4IP
- `10_raman_models.jl` - Compare Blow-Wood, Lin-Agrawal, Hollenbeck

## Contributing Examples

When adding examples:
1. Follow descriptive naming: `feature_name.jl`
2. Include comments explaining physics and parameters
3. Add entry to this README
4. Keep runtime reasonable (< 1 minute preferred)
5. Generate informative visualizations
6. Use realistic physical parameters
7. Use Plots.jl for consistency and compatibility
