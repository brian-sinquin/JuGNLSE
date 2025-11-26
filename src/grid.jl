"""
    create_grid(N::Int, time_window::Real, center_wavelength::Real)

Create a time-frequency grid for GNLSE simulations.

# Arguments
- `N::Int`: Number of grid points (should be power of 2 for FFT efficiency)
- `time_window::Real`: Total time window [s]
- `center_wavelength::Real`: Center wavelength [m]

# Returns
- `Grid`: Grid structure containing time and frequency arrays

# Example
```julia
grid = create_grid(2^12, 10e-12, 835e-9)  # 4096 points, 10 ps window, 835 nm
```
"""
function create_grid(N::Int, time_window::Real, center_wavelength::Real)
    N > 0 || throw(ArgumentError("N must be positive"))
    ispow2(N) || @warn "N should be a power of 2 for optimal FFT performance"
    time_window > 0 || throw(ArgumentError("time_window must be positive"))
    center_wavelength > 0 || throw(ArgumentError("center_wavelength must be positive"))
    
    # Time grid
    dt = time_window / N
    t = range(-time_window/2, time_window/2, length=N) |> collect
    
    # Frequency grid (angular frequency)
    domega = 2π / time_window
    omega_max = π / dt
    omega = range(-omega_max, stop=omega_max-domega, length=N) |> collect
    
    # Apply fftshift to get FFT order: [0, dω, 2dω, ..., ωmax, -ωmax, ..., -dω]
    omega = fftshift(omega)
    
    Grid(N, t, omega, dt, domega)
end

"""
    create_grid_from_medium(N::Int, medium::Medium; time_window::Real=nothing)

Create a grid based on medium parameters. Automatically estimates appropriate time window
if not provided based on dispersion length and nonlinear length.

# Arguments
- `N::Int`: Number of grid points
- `medium::Medium`: Medium parameters
- `time_window::Real`: (Optional) Time window [s]. If not provided, estimated automatically.

# Returns
- `Grid`: Grid structure
"""
function create_grid_from_medium(N::Int, medium::Medium; time_window::Real=nothing)
    if time_window === nothing
        # Estimate reasonable time window (heuristic)
        # Use approximately 10 times the expected pulse duration
        time_window = 20e-12  # 20 ps default
    end
    
    create_grid(N, time_window, medium.lambda0)
end
