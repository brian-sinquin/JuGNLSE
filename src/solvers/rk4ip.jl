"""
RK4IP solver: Fourth-order Runge-Kutta in the Interaction Picture method.
Reference: J. Hult, J. Lightwave Tech. 25, 3770-3775 (2007)

This is a fixed-step size variant of the adaptive ERK4IP solver, using only
4 RK stages without embedded error estimation. Simpler than ERK4IP but requires
manual step size selection.
"""

using FFTW
using LinearAlgebra: mul!

# Import nonlinearity module functions
import ..build_physics_model, ..PhysicsModel

"""
    propagate_rk4ip(pulse::Pulse, params::SimParams; progress::Bool=true, n_steps::Int=1000)

Propagate pulse using 4th-order Runge-Kutta in Interaction Picture (RK4IP).

The RK4IP method from Hult (2007) splits the propagation into linear (dispersion)
and nonlinear parts using the interaction picture transformation. Uses fixed step
size unlike the adaptive ERK4IP solver.

# Algorithm

For each step of size dz:

 1. Transform to interaction picture: Âᵢₚ = exp(D̂·dz/2)·Â

 2. Compute 4 RK stages:

      + k₁ = exp(D̂·dz/2)·N̂[A(z)]
      + k₂ = N̂[IFFT(Âᵢₚ + k₁/2)]
      + k₃ = N̂[IFFT(Âᵢₚ + k₂/2)]
      + k₄ = N̂[IFFT(exp(D̂·dz/2)·(Âᵢₚ + k₃))]
 3. Update: Â(z+dz) = exp(D̂·dz/2)·(Âᵢₚ + (k₁ + 2k₂ + 2k₃)/6) + k₄/6

where:

  - D̂ is the linear dispersion operator
  - N̂[A] is the nonlinear operator (Kerr, Raman, shock)
  - exp(D̂·dz/2) advances dispersion by half-step

# Arguments

  - `pulse::Pulse`: Initial pulse with grid and fields
  - `params::SimParams`: Simulation parameters including medium properties

# Keyword Arguments

  - `progress::Bool=true`: Print progress messages
  - `n_steps::Int=1000`: Number of propagation steps (fixed step size = length/n_steps)

# Returns

  - `z_out`: Array of z positions [m]
  - `At_out`: Time-domain field at each save point [√W]
  - `Aw_out`: Frequency-domain field at each save point [√W·s]

# Notes

  - Fixed step size dz = medium.length / n_steps
  - No adaptive stepping - user must choose n_steps carefully
  - For adaptive stepping, use `propagate_erk4ip` instead
  - Step size should satisfy: dz << Lₙₗ, Lᴅ for accuracy

# Example

```julia
grid = create_grid(2^12, 10e-12, 835e-9)
medium = Medium(0.15, 0.11, [-11.83e-27, 8.11e-41], 0.0, 835e-9)
pulse = sech_pulse(grid, 50e-15, 10000.0)
params = SimParams(; medium=medium, n_saves=200, raman=true, shock=true)

# Fixed step size: 150mm / 5000 steps equals 30μm per step
z, At, Aw = propagate_rk4ip(pulse, params; n_steps=5000)
```

# See Also

  - [`propagate_erk4ip`](@ref): Adaptive variant with embedded error estimation
  - [`propagate_ssfm`](@ref): Simpler split-step Fourier method

# Reference

J. Hult, "A Fourth-Order Runge–Kutta in the Interaction Picture Method for
Simulating Supercontinuum Generation in Optical Fibers," J. Lightwave Technol.
25(12), 3770-3775 (2007). DOI: 10.1109/JLT.2007.909373
"""
function propagate_rk4ip(
    pulse::Pulse,
    params::SimParams;
    progress::Bool=true,
    n_steps::Union{Int, Nothing}=nothing,
    dz::Union{Float64, Nothing}=nothing,
)
    grid = pulse.grid
    medium = params.medium
    N = grid.N
    z_end = medium.length
    n_saves = params.n_saves

    # Determine n_steps from dz if provided
    if dz !== nothing
        n_steps = round(Int, z_end / dz)
    elseif n_steps === nothing
        # Use params.dz as default
        n_steps = round(Int, z_end / params.dz)
    end

    # Build physics model once
    model = build_physics_model(grid, params)

    # Initial condition in frequency domain
    U = copy(pulse.Aw)

    # Fixed step size
    dz = z_end / n_steps

    # Storage for output
    z_out = zeros(n_saves)
    At_out = zeros(ComplexF64, N, n_saves)
    Aw_out = zeros(ComplexF64, N, n_saves)

    # Save initial condition
    z_out[1] = 0.0
    At_out[:, 1] .= pulse.At
    Aw_out[:, 1] .= U

    # Determine save interval
    save_interval = n_steps ÷ (n_saves - 1)
    z = 0.0
    save_idx = 2

    # Pre-allocate workspace
    e = similar(U)              # exp(D̂·dz/2)
    Uip = similar(U)            # U in interaction picture
    k1 = similar(U)             # RK stage 1
    k2 = similar(U)             # RK stage 2
    k3 = similar(U)             # RK stage 3
    k4 = similar(U)             # RK stage 4
    NU = similar(U)             # Nonlinear operator N̂[U]
    At_temp = similar(pulse.At) # Time-domain buffer
    Aw_temp = similar(U)        # Frequency-domain buffer

    # Initial nonlinearity: N̂[A(z=0)]
    mul!(At_temp, model.ifftp, U)
    NU .= model.nonlinear_function(At_temp, model)

    # Linear half-step operator (constant for fixed dz)
    @. e = exp(0.5 * dz * model.dispersion_term)

    # Initialize progress bar
    prog = if progress
        Progress(n_steps; desc="RK4IP (dz=$(round(dz*1e6, digits=1))μm): ", showspeed=true)
    else
        nothing
    end

    # Main propagation loop
    for step in 1:n_steps
        # ============================================================
        # RK4IP Method - Hult (2007) Algorithm
        # ============================================================

        # Transform to interaction picture: Ûᵢₚ = exp(D̂·dz/2)·U
        @. Uip = e * U

        # Stage 1: k₁ = exp(D̂·dz/2)·N̂[A(z)]
        @. k1 = e * NU

        # Stage 2: k₂ = N̂[IFFT(Ûᵢₚ + dz·k₁/2)]
        @. Aw_temp = Uip + 0.5 * dz * k1
        mul!(At_temp, model.ifftp, Aw_temp)
        k2 .= model.nonlinear_function(At_temp, model)

        # Stage 3: k₃ = N̂[IFFT(Ûᵢₚ + dz·k₂/2)]
        @. Aw_temp = Uip + 0.5 * dz * k2
        mul!(At_temp, model.ifftp, Aw_temp)
        k3 .= model.nonlinear_function(At_temp, model)

        # Stage 4: k₄ = N̂[IFFT(exp(D̂·dz/2)·(Ûᵢₚ + dz·k₃))]
        @. Aw_temp = e * (Uip + dz * k3)
        mul!(At_temp, model.ifftp, Aw_temp)
        k4 .= model.nonlinear_function(At_temp, model)

        # Update solution: U(z+dz) = exp(D̂·dz/2)·(Ûᵢₚ + dz·(k₁ + 2k₂ + 2k₃)/6) + dz·k₄/6
        @. U = e * (Uip + dz * (k1 + 2.0 * k2 + 2.0 * k3) / 6.0) + dz * k4 / 6.0

        # Update nonlinearity for next step (FSAL property)
        mul!(At_temp, model.ifftp, U)
        NU .= model.nonlinear_function(At_temp, model)

        # Advance position
        z += dz

        # Update progress bar
        if !isnothing(prog)
            update!(prog, step)
        end

        # Save if at save point
        if step % save_interval == 0 && save_idx <= n_saves
            z_out[save_idx] = z
            copyto!(@view(Aw_out[:, save_idx]), U)
            # Transform to time domain and shift to natural order
            mul!(At_temp, model.ifftp, U)
            At_out[:, save_idx] .= fftshift(At_temp)
            save_idx += 1
        end
    end

    # Ensure final point is saved
    if save_idx <= n_saves
        z_out[save_idx] = z_end
        copyto!(@view(Aw_out[:, save_idx]), U)
        # Transform to time domain and shift to natural order
        mul!(At_temp, model.ifftp, U)
        At_out[:, save_idx] .= fftshift(At_temp)
    end

    if progress
        println("RK4IP propagation complete")
    end

    return z_out, At_out, Aw_out
end
