"""
    solve(pulse::Pulse, params::SimParams; method::Symbol=:rk4ip, progress::Bool=true, 
          rtol::Float64=1e-6, atol::Float64=1e-8)

Main solver interface for GNLSE propagation.

# Arguments
- `pulse::Pulse`: Initial pulse
- `params::SimParams`: Simulation parameters
- `method::Symbol`: Integration method: :rk4ip, :erk4ip, or :ssfm (default: :rk4ip)
- `progress::Bool`: Show progress information (default: true)
- `rtol::Float64`: Relative tolerance for :erk4ip adaptive stepping (default: 1e-6)
- `atol::Float64`: Absolute tolerance for :erk4ip adaptive stepping (default: 1e-8)

# Returns
- `NamedTuple`: Results with fields (z, At, Aw, grid, params)

# Methods
- `:rk4ip` - Runge-Kutta 4th order in Interaction Picture (fixed step, high accuracy)
- `:erk4ip` - Embedded RK4(3) in Interaction Picture (adaptive step, error control)
- `:ssfm` - Split-Step Fourier Method (fixed step, fast but less accurate)

# Example
```julia
# Use RK4IP with fixed steps
results = solve(pulse, params, method=:rk4ip)

# Use ERK4IP with adaptive stepping
results = solve(pulse, params, method=:erk4ip, rtol=1e-7)

# Use SSFM
results = solve(pulse, params, method=:ssfm)
```
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
