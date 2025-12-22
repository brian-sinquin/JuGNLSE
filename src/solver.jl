"""
    solve(pulse::Pulse, params::SimParams; method::Symbol=:ERK4IP,
          progress::Bool=true, rtol::Float64=1e-6, atol::Float64=1e-8,
          dz::Union{Float64,Nothing}=nothing, adaptive::Bool=false, n_steps::Int=1000)

Main solver interface for GNLSE propagation in nonlinear optical fibers.

Solves the Generalized Nonlinear Schrödinger Equation:

```
∂A/∂z = D̂[A] + N̂[A]
```

where D̂ is the dispersion operator and N̂ is the nonlinear operator.

# Arguments

  - `pulse::Pulse`: Initial pulse (time and frequency domain representations)
  - `params::SimParams`: Simulation parameters (medium, grid size, effects, tolerances)
  - `method::Symbol`: Solver method - `:ERK4IP` (default), `:RK4IP`, or `:SSFM`
  - `progress::Bool`: Show progress information during propagation (default: `true`)
  - `rtol::Float64`: Relative tolerance for adaptive solvers (ERK4IP) (default: 1e-6)
  - `atol::Float64`: Absolute tolerance for adaptive solvers (ERK4IP) (default: 1e-8)
  - `dz::Union{Float64,Nothing}`: Fixed step size for SSFM [m] (auto-computed if `nothing`)
  - `adaptive::Bool`: Use adaptive stepping for SSFM (experimental, default: false)
  - `n_steps::Int`: Number of steps for RK4IP fixed-step solver (default: 1000)

# Returns

  - `NamedTuple` with fields:

      + `z::Vector{Float64}`: Propagation distances [m]
      + `At::Matrix{ComplexF64}`: Time-domain field evolution (grid points × save points)
      + `Aw::Matrix{ComplexF64}`: Frequency-domain field evolution
      + `grid::Grid`: Time-frequency grid used in simulation
      + `params::SimParams`: Parameters used (for reference)

# Solver Methods

## ERK4IP - Embedded Runge-Kutta 4th order in Interaction Picture (Default)

  - **Accuracy**: 4th/5th order embedded pair for error estimation
  - **Step control**: Adaptive stepping with user-specified tolerances
  - **Speed**: Optimized for Julia with pre-allocated buffers
  - **Features**: Automatic step size adjustment, interaction picture formulation
  - **Typical tolerances**: `rtol=1e-6`, `atol=1e-8`
  - **Best for**: General use, high accuracy with fewer steps

## RK4IP - Runge-Kutta 4th order in Interaction Picture

  - **Accuracy**: 4th order (no error estimation)
  - **Step control**: Fixed step size (user specifies n_steps)
  - **Speed**: Simpler than ERK4IP, faster per-step
  - **Features**: Classic RK4IP from Hult (2007), no adaptive control
  - **Typical steps**: `n_steps = 1000-5000` depending on nonlinearity
  - **Best for**: Comparison with fixed-step references, benchmarking

## SSFM - Symmetric Split-Step Fourier Method

  - **Accuracy**: 2nd order (symmetric split)
  - **Step control**: Fixed step (default) or adaptive (experimental)
  - **Speed**: Simpler per-step but requires more steps than RK4IP/ERK4IP
  - **Features**: Classic, robust method; good for validation
  - **Typical step**: `dz = L / (20 * n_saves)` (auto-computed)
  - **Best for**: Validation, benchmarking, very long propagation

# Default: ERK4IP (adaptive, high accuracy)

results = solve(pulse, params)

# Alternative: RK4IP (fixed step, 4th order)

results = solve(pulse, params, method=:RK4IP, n_steps=2000)

# Alternative: SSFM (fixed step, robust)

results = solve(pulse, params, method=:SSFM)

# Custom SSFM step size

results = solve(pulse, params, method=:SSFM, dz=1e-4)

```julia
grid = create_grid(2^12, 10e-12, 835e-9)
pulse = sech_pulse(grid, 50e-15, 10000.0)
medium = Medium(0.15, 0.11, [-11.83e-27], 0.0, 835e-9)
params = SimParams(; medium=medium, n_saves=200)

results = solve(pulse, params)
# results.z:  [0.0, 0.00075, ..., 0.15] (200 points)
# results.At: complex field evolution
```

## High-Accuracy Soliton Simulation

```julia
# Calculate N=1 soliton parameters
beta2 = -11.83e-27
gamma = 0.11
T0 = 28.4e-15  # 1/e half-width
P0 = abs(beta2) / (gamma * T0^2)

pulse = sech_pulse(grid, T0, P0; T0=true)
medium = Medium(1.0, gamma, [beta2], 0.0, 835e-9)  # 1 m fiber
params = SimParams(; medium=medium, raman=false, shock=false)

# Tight tolerances for soliton fidelity check
results = solve(pulse, params; rtol=1e-8, atol=1e-10)
# Should maintain shape over multiple soliton periods
```

## Supercontinuum Generation

```julia
# Short pulse in anomalous dispersion with all effects
pulse = sech_pulse(grid, 28.4e-15, P0 * 5; T0=true)  # N=5 soliton
params = SimParams(;
    medium=medium,
    N=2^14,          # High resolution for broad spectrum
    n_saves=500,     # Fine z-resolution
    raman=true,      # Essential for SC
    shock=true,      # Essential for short pulses
    reltol=1e-7,      # Tight for nonlinear dynamics
)

results = solve(pulse, params)
# Rich spectral dynamics: fission, Raman solitons, dispersive waves
```

## Parameter Sweep

```julia
powers = range(1000, 20000; length=20)
results_sweep = map(powers) do P
    pulse = sech_pulse(grid, 50e-15, P, 835e-9)
    results = solve(pulse, params; progress=false)
    (P=P, energy_out=sum(abs2.(results.At[:, end])) * grid.dt)
end
# Fast exploration of parameter space
```

# Performance Tips

 1. **Grid size**: Use power of 2 (2^12, 2^14) for FFT efficiency
 2. **Time window**: 10× pulse duration to avoid wrap-around
 3. **Tolerances**: Tighter (`rtol=1e-8`) = slower but more accurate, looser (`rtol=1e-5`) = faster
 4. **Save points**: More `n_saves` → more memory, finer z-resolution
 5. **Adaptive stepping**: ERK4IP automatically adjusts step size for optimal performance

# Notes on Accuracy

-olver Comparison

| Feature               | ERK4IP           | RK4IP           | SSFM           |
|:--------------------- |:---------------- |:--------------- |:-------------- |
| **Order of accuracy** | 4th/5th          | 4th             | 2nd            |
| **Steps required**    | Fewer (adaptive) | Medium (fixed)  | More (fixed)   |
| **Per-step cost**     | Higher (10 FFTs) | Medium (7 FFTs) | Lower (3 FFTs) |
| **Robustness**        | Excellent        | Excellent       | Excellent      |
| **Best for**          | General use      | Benchmarks      | Validation     |

# See Also

  - [`propagate_erk4ip`](@ref): ERK4IP implementation details
  - [`propagate_rk4ip`](@ref): RK4IP implementation details
  - [`propagate_ssfm`](@ref): SSFM implementation details
  - [`SimParams`](@ref): Parameter structure documentation
"""
function solve(
    pulse::Pulse,
    params::SimParams;
    method::Symbol=:ERK4IP,
    progress::Bool=true,
    rtol::Float64=1e-6,
    atol::Float64=1e-8,
    dz::Union{Float64, Nothing}=nothing,
    adaptive::Bool=false,
    n_steps::Int=1000,
)
    # Dispatch to appropriate solver
    if method == :ERK4IP
        z, At, Aw = propagate_erk4ip(
            pulse, params; progress=progress, rtol=rtol, atol=atol, dz=dz
        )
    elseif method == :RK4IP
        z, At, Aw = propagate_rk4ip(
            pulse, params; progress=progress, n_steps=n_steps, dz=dz
        )
    elseif method == :SSFM
        z, At, Aw = propagate_ssfm(
            pulse, params; progress=progress, adaptive=adaptive, dz=dz
        )
    else
        error("Unknown solver method: $method. Use :ERK4IP, :RK4IP, or :SSFM")
    end

    return (z=z, At=At, Aw=Aw, grid=pulse.grid, params=params, method=method)
end
