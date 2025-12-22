"""
    create_grid(N::Int, time_window::Real, center_wavelength::Real)

Create time-frequency grid for GNLSE simulations.

# Arguments

  - `N::Int`: Number of grid points (power of 2 recommended for FFT efficiency)
  - `time_window::Real`: Total time window [s]
  - `center_wavelength::Real`: Center wavelength [m]

# Returns

  - `Grid`: Grid structure with time domain symmetric around t=0 spanning [-time_window/2, time_window/2), frequency domain as angular frequency detuning Δω = ω - ω₀ [rad/s], and resolution dt = time_window/N, dω = 2π/time_window.
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

Create grid using medium parameters with automatic time window estimation.

# Arguments

  - `N::Int`: Number of grid points (power of 2 recommended)
  - `medium::Medium`: Medium parameters (extracts lambda0 for center frequency)
  - `time_window::Real`: Time window [s], defaults to 20 ps if not specified

# Returns

  - `Grid`: Grid structure configured for the specified medium
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
