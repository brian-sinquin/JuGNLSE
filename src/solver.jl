"""
    solve(pulse::Pulse, params::SimParams; method=:ERK4IP, rtol=1e-6, at ol=1e-8, kwargs...)

Solve the generalized nonlinear Schrödinger equation for pulse propagation in optical fibers.

Equation: ∂A/∂z = D̂[A] + N̂[A]

# Parameters

  - `pulse`: Initial pulse condition
  - `params`: Simulation parameters (medium, effects, save points)
  - `method`: Solver choice - `:ERK4IP` (adaptive, default), `:RK4IP` (fixed), `:SSFM`
  - `rtol`, `atol`: Error tolerances for adaptive methods
  - `dz`: Step size (auto-selected if `nothing`)
  - `n_steps`: Number of steps for fixed-step methods

# Returns

NamedTuple with `z`, `At`, `Aw`, `grid`, `params`, `method`

# Example

```julia
grid = create_grid(2^12, 10e-12, 835e-9)
pulse = sech_pulse(grid, 50e-15, 10000.0)
medium = Medium(0.15, 0.11, [-11.83e-27], 0.0, 835e-9)
params = SimParams(; medium=medium, n_saves=200)
results = solve(pulse, params)
```

# Solver Comparison

  - **ERK4IP**: 4th order adaptive, recommended for general use
  - **RK4IP**: 4th order fixed-step, for benchmarking
  - **SSFM**: 2nd order, robust classical method

See [`propagate_erk4ip`](@ref), [`propagate_rk4ip`](@ref), [`propagate_ssfm`](@ref)
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
