"""
    solve(pulse::Pulse, params::SimParams; method::Symbol=:rk4ip, progress::Bool=true, 
          rtol::Float64=1e-6, atol::Float64=1e-8)

Main solver interface for GNLSE propagation in nonlinear optical fibers.

Solves the Generalized Nonlinear Schrödinger Equation:
```
∂A/∂z = D̂[A] + N̂[A]
```
where D̂ is the dispersion operator and N̂ is the nonlinear operator.

# Arguments
- `pulse::Pulse`: Initial pulse (time and frequency domain representations)
- `params::SimParams`: Simulation parameters (medium, grid size, effects, tolerances)
- `method::Symbol`: Integration method - `:rk4ip`, `:erk4ip`, or `:ssfm` (default: `:rk4ip`)
- `progress::Bool`: Show progress information during propagation (default: `true`)
- `rtol::Float64`: Relative tolerance for adaptive methods (`:erk4ip`) (default: 1e-6)
- `atol::Float64`: Absolute tolerance for adaptive methods (`:erk4ip`) (default: 1e-8)

# Returns
- `NamedTuple` with fields:
  * `z::Vector{Float64}`: Propagation distances [m]
  * `At::Matrix{ComplexF64}`: Time-domain field evolution (grid points × save points)
  * `Aw::Matrix{ComplexF64}`: Frequency-domain field evolution
  * `grid::Grid`: Time-frequency grid used in simulation
  * `params::SimParams`: Parameters used (for reference)

# Solver Methods

## :rk4ip (Recommended for accuracy)
**Runge-Kutta 4th order in Interaction Picture**
- **Accuracy**: 4th order, very high precision
- **Step control**: Adaptive via OrdinaryDiffEq.jl (DP5 algorithm)
- **Speed**: Moderate (more function evaluations than SSFM)
- **Use when**: Accuracy is critical, studying subtle effects (Raman, shock), benchmarking
- **Typical tolerances**: `reltol=1e-6`, `abstol=1e-9`

## :erk4ip (Recommended for efficiency)
**Embedded RK4(3) in Interaction Picture**
- **Accuracy**: 4th order with 3rd order error estimation
- **Step control**: Custom adaptive stepping (Bogacki-Shampine coefficients)
- **Speed**: Fast with automatic step adaptation
- **Use when**: Long propagation distances, parameter sweeps, need error control
- **Typical tolerances**: `reltol=1e-6` to `1e-8`
- **Advantage**: FSAL property (First Same As Last) reduces function evaluations

## :ssfm (Recommended for speed)
**Split-Step Fourier Method**
- **Accuracy**: 2nd order (Strang splitting)
- **Step control**: Fixed step size (`dz = length/(n_saves-1)`)
- **Speed**: Fastest (only 2 FFTs per step, no iterations)
- **Use when**: Quick previews, well-understood systems, speed over precision
- **Limitation**: No automatic error control - accuracy depends on `n_saves`

# Method Comparison Table
```
Method    | Order | Adaptive | Speed   | Accuracy | Best for
----------|-------|----------|---------|----------|------------------
:rk4ip    |  4th  |   Yes    | Slow    | Highest  | Benchmarks, validation
:erk4ip   |  4th  |   Yes    | Fast    | High     | Production simulations
:ssfm     |  2nd  |   No     | Fastest | Moderate | Quick studies, previews
```

# Examples

## Basic Usage (RK4IP - highest accuracy)
```julia
grid = create_grid(2^12, 10e-12, 835e-9)
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
medium = Medium(0.15, 0.11, [-11.83e-27], 0.0, 835e-9)
params = SimParams(medium=medium, n_saves=200)

results = solve(pulse, params)  # Uses :rk4ip by default
# results.z:  [0.0, 0.00075, ..., 0.15] (200 points)
# results.At: complex field evolution
```

## High-Accuracy Soliton Simulation (ERK4IP with tight tolerances)
```julia
# Calculate N=1 soliton parameters
beta2 = -11.83e-27
gamma = 0.11
T0 = 28.4e-15  # 1/e half-width
P0 = abs(beta2) / (gamma * T0^2)

pulse = sech_pulse(grid, T0, P0, 835e-9, T0=true)
medium = Medium(1.0, gamma, [beta2], 0.0, 835e-9)  # 1 m fiber
params = SimParams(medium=medium, raman=false, shock=false)

# Tight tolerances for soliton fidelity check
results = solve(pulse, params, method=:erk4ip, rtol=1e-8, atol=1e-10)
# Should maintain shape over multiple soliton periods
```

## Fast Preview with SSFM
```julia
params = SimParams(medium=medium, n_saves=100)  # Fewer saves = larger steps
results = solve(pulse, params, method=:ssfm, progress=false)
# Fast visualization of propagation dynamics
```

## Supercontinuum Generation (ERK4IP recommended)
```julia
# Short pulse in anomalous dispersion with all effects
pulse = sech_pulse(grid, 28.4e-15, P0*5, 835e-9, T0=true)  # N=5 soliton
params = SimParams(
    medium = medium,
    N = 2^14,          # High resolution for broad spectrum
    n_saves = 500,     # Fine z-resolution
    raman = true,      # Essential for SC
    shock = true,      # Essential for short pulses
    raman_model = Hollenbeck(),  # Most accurate
    reltol = 1e-7      # Tight for nonlinear dynamics
)

results = solve(pulse, params, method=:erk4ip)
# Rich spectral dynamics: fission, Raman solitons, dispersive waves
```

## Parameter Sweep (SSFM for speed)
```julia
powers = range(1000, 20000, length=20)
results_sweep = map(powers) do P
    pulse = sech_pulse(grid, 50e-15, P, 835e-9)
    results = solve(pulse, params, method=:ssfm, progress=false)
    (P=P, energy_out=sum(abs2.(results.At[:, end])) * grid.dt)
end
# Fast exploration of parameter space
```

# Performance Tips
1. **Grid size**: Use power of 2 (2^12, 2^14) for FFT efficiency
2. **Time window**: 10× pulse duration to avoid wrap-around
3. **Method selection**:
   - Exploring → use `:ssfm` with `n_saves=100-200`
   - Production → use `:erk4ip` with `reltol=1e-6`
   - Validation → use `:rk4ip` with `reltol=1e-8`
4. **Tolerances**: Tighter = slower but more accurate
5. **Save points**: More `n_saves` → more memory, finer z-resolution

# Notes on Accuracy
- **Soliton test**: N=1 soliton should maintain 99.999% fidelity over multiple soliton periods
- **Energy conservation**: Should conserve to within tolerance (losses from α excepted)
- **Convergence**: Halving step size (double `n_saves` for SSFM) should yield same result
- **Spectral resolution**: Ensure `time_window` captures all spectral broadening

# See Also
- [`propagate_rk4ip`](@ref): RK4IP implementation details
- [`propagate_erk4ip`](@ref): ERK4IP implementation details
- [`propagate_ssfm`](@ref): SSFM implementation details
- [`SimParams`](@ref): Parameter structure documentation
"""
function solve(pulse::Pulse, params::SimParams; method::Symbol=:rk4ip, progress::Bool=true,
               rtol::Float64=1e-6, atol::Float64=1e-8)
    if method == :ssfm
        z, At, Aw = propagate_ssfm(pulse, params, progress=progress)
    elseif method == :rk4ip
        z, At, Aw = propagate_rk4ip(pulse, params, progress=progress)
    elseif method == :erk4ip
        z, At, Aw = propagate_erk4ip(pulse, params, progress=progress, rtol=rtol, atol=atol)
    else
        throw(ArgumentError("Unknown method: $method. Use :rk4ip, :erk4ip, or :ssfm"))
    end
    
    (z=z, At=At, Aw=Aw, grid=pulse.grid, params=params)
end
