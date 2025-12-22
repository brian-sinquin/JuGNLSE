"""
    create_grid(N::Int, time_window::Real, center_wavelength::Real)

Create a time-frequency grid for GNLSE simulations.

# Arguments

  - `N::Int`: Number of grid points (strongly recommend power of 2 for FFT efficiency)
  - `time_window::Real`: Total time window [s]
  - `center_wavelength::Real`: Center wavelength [m]

# Returns

  - `Grid`: Grid structure containing time and frequency arrays

# Grid Properties

  - **Time domain**: Symmetric around t=0, spans `[-time_window/2, time_window/2)`
  - **Frequency domain**: Angular frequency detuning Δω = ω - ω₀ in rad/s
  - **FFT convention**: Pre-arranged with `fftshift` so `ifft(A(t)) → A(ω)` and `fft(A(ω)) → A(t)`
  - **Resolution**: `dt = time_window/N`, `dω = 2π/time_window`

# Examples

```julia
# Standard grid for femtosecond pulse simulation
grid = create_grid(2^12, 10e-12, 835e-9)
# 4096 points, 10 ps window, λ₀ = 835 nm
# dt = 2.44 fs, dω = 0.628 rad/fs

# High-resolution grid for broadband supercontinuum
grid = create_grid(2^14, 20e-12, 1030e-9)
# 16384 points, 20 ps window, λ₀ = 1030 nm
# dt = 1.22 fs, dω = 0.314 rad/fs
```

# Notes

  - Larger `N` provides better frequency resolution but increases memory and computation time
  - `time_window` should be ~10× the pulse duration to avoid wrap-around effects
  - Non-power-of-2 `N` will work but FFT performance may be significantly degraded

# See Also

  - [`create_grid_from_medium`](@ref): Automatically estimate time window from medium
  - [`Grid`](@ref): Grid structure documentation
"""
function create_grid(N::Int, time_window::Real, center_wavelength::Real)
    N > 0 || throw(ArgumentError("N must be positive"))
    ispow2(N) || @warn "N should be a power of 2 for optimal FFT performance"
    time_window > 0 || throw(ArgumentError("time_window must be positive"))
    center_wavelength > 0 || throw(ArgumentError("center_wavelength must be positive"))

    # Time grid
    dt = time_window / N
    t = dt * ((-N ÷ 2):(N ÷ 2 - 1)) |> collect  # Symmetric around zero

    # Frequency grid (angular frequency detuning Δω = ω - ω₀)
    # This matches the output of ifft (time→freq in JuGNLSE's inverted convention)
    omega = 2π .* fftshift(((-N ÷ 2):(N ÷ 2 - 1)) ./ time_window)  # Converts frequency [Hz] to angular frequency [rad/s]

    # Central frequency and wavelength
    omega0 = 2π * SPEED_OF_LIGHT / center_wavelength  # Central angular frequency [rad/s]
    lambda0 = center_wavelength  # Center wavelength [m]

    # Wavelength grid: λ = 2πc/(ω + ω₀) = c/(ν₀ + Δν)
    lambda = 2π * SPEED_OF_LIGHT ./ (omega .+ omega0)  # Wavelength [m]

    Grid(N, t, omega, dt, omega0, lambda0, lambda)
end

"""
    create_grid_from_medium(N::Int, medium::Medium; time_window::Real=nothing)

Create a grid based on medium parameters with automatic time window estimation.

# Arguments

  - `N::Int`: Number of grid points (power of 2 recommended)
  - `medium::Medium`: Medium parameters (uses lambda0 for center frequency)
  - `time_window::Real`: (Optional) Time window [s]. If `nothing`, uses default 20 ps.

# Returns

  - `Grid`: Grid structure configured for the given medium

# Examples

```julia
# Automatic time window (20 ps default)
grid = create_grid_from_medium(2^12, medium)

# Custom time window
grid = create_grid_from_medium(2^12, medium; time_window=50e-12)  # 50 ps
```

# Notes

  - Default 20 ps window is suitable for most femtosecond pulse simulations
  - For picosecond pulses or long propagation, increase `time_window`
  - For sub-10 fs pulses, consider reducing to 5-10 ps to save memory

# See Also

        # Estimate reasonable time window (heuristic)

  - [`create_grid`](@ref): Direct grid creation with explicit parameters        # Use approximately 10 times the expected pulse duration
"""
function create_grid_from_medium(
    N::Int, medium::Medium; time_window::Union{Real, Nothing}=nothing
)
    if time_window === nothing
        # Estimate reasonable time window (heuristic)
        # Use approximately 10 times the expected pulse duration
        time_window = 20e-12  # 20 ps default
    end

    create_grid(N, time_window, medium.lambda0)
end
