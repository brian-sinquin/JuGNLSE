"""
ERK4IP solver following FiberNlse's clean algorithm structure.
Uses PhysicsModel for precomputed operators and FSAL property.
"""

using FFTW
using LinearAlgebra: mul!
using ProgressMeter: Progress, update!

# Import nonlinearity module functions
import ..build_physics_model, ..PhysicsModel

"""
    propagate_erk4ip(pulse::Pulse, params::SimParams; progress=true, rtol=1e-6, atol=1e-8, dz=nothing)

Adaptive embedded Runge-Kutta 4(3) solver in interaction picture for the GNLSE.

This is the recommended solver for most simulations, providing adaptive step size control
with embedded error estimation while maintaining high accuracy and efficiency.

# Algorithm

Uses a 4th-order Runge-Kutta method with 3rd-order embedded error estimator (ERK4IP)
in the interaction picture. The interaction picture transformation removes the linear
dispersion operator, making the nonlinear evolution easier to integrate accurately.

# Parameters

  - `pulse::Pulse`: Initial pulse with time and frequency domain fields
  - `params::SimParams`: Simulation parameters including medium properties
  - `progress::Bool=true`: Show progress bar during propagation
  - `rtol::Float64=1e-6`: Relative error tolerance for adaptive stepping
  - `atol::Float64=1e-8`: Absolute error tolerance for adaptive stepping
  - `dz::Union{Float64, Nothing}=nothing`: Initial step size [m], auto-selected if `nothing`

# Returns

  - `z::Vector{Float64}`: Propagation distances [m]
  - `At::Matrix{ComplexF64}`: Time-domain field at each save point
  - `Aw::Matrix{ComplexF64}`: Frequency-domain field at each save point

# Physics

Solves the GNLSE in the interaction picture:

```math
\\frac{\\partial U}{\\partial z} = e^{-\\hat{D}z} \\hat{N}[e^{\\hat{D}z}U]
```

where ``\\hat{D}`` is the linear dispersion operator and ``\\hat{N}`` includes
Kerr nonlinearity, Raman scattering, and self-steepening (shock) effects.

# Examples

```julia
grid = create_grid(2^12, 10e-12, 835e-9)
medium = Medium(1.0, 0.1, [-11.8e-27, 8.1e-41], 0.0, 835e-9)
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
params = SimParams(; medium=medium, n_saves=200, raman=true, shock=true)

z, At, Aw = propagate_erk4ip(pulse, params; rtol=1e-6)
```

# See Also

  - [`propagate_rk4ip`](@ref): Fixed-step RK4IP for comparison
  - [`propagate_ssfm`](@ref): Classical split-step Fourier method
  - [`solve`](@ref): High-level interface that selects this solver by default

# References

Heidt, A. M. (2009). Efficient Adaptive Step Size Method for the Simulation of
Supercontinuum Generation in Optical Fibers. *Journal of Lightwave Technology*, 27(18).
"""
function propagate_erk4ip(
    pulse::Pulse,
    params::SimParams;
    progress::Bool=true,
    rtol::Float64=1e-6,
    atol::Float64=1e-8,
    dz::Union{Float64, Nothing}=nothing,
)
    grid = pulse.grid
    medium = params.medium
    N = grid.N
    z_end = medium.length
    n_saves = params.n_saves

    # Build physics model once (FiberNlse approach)
    model = build_physics_model(grid, params)

    # Initial condition in frequency domain
    U = copy(pulse.Aw)

    # Storage
    z_out = zeros(n_saves)
    At_out = zeros(ComplexF64, N, n_saves)
    Aw_out = zeros(ComplexF64, N, n_saves)

    z_out[1] = 0.0
    At_out[:, 1] .= pulse.At
    Aw_out[:, 1] .= U

    # Step size control
    z = 0.0
    dz = dz === nothing ? params.dz : dz  # Use user-provided initial step size
    save_idx = 2
    z_saves = range(0, z_end; length=n_saves)

    # Pre-allocate workspace following FiberNlse naming
    e = similar(U)              # exp(D̂·dz/2)
    Uip = similar(U)            # U in interaction picture
    k1 = similar(U)
    k2 = similar(U)
    k3 = similar(U)
    k4 = similar(U)
    k5 = similar(U)
    r = similar(U)              # RK accumulator
    U1 = similar(U)             # 4th-order solution
    U2 = similar(U)             # 5th-order solution
    NU = similar(U)             # Nonlinear operator N̂[U]
    At_temp = similar(pulse.At)  # Time-domain buffer
    Aw_temp = similar(U)          # Frequency-domain buffer

    # Initial nonlinearity: N̂[A(z=0)]
    mul!(At_temp, model.ifftp, U)
    NU .= model.nonlinear_function(At_temp, model)

    # Initialize progress bar
    prog = progress ? Progress(n_saves - 1; desc="ERK4IP: ", showspeed=true) : nothing

    step_count = 0
    while z < z_end && save_idx <= n_saves
        step_count += 1

        # Limit step to reach next save point
        z_target = z_saves[save_idx]
        dz = min(dz, z_target - z)

        # ============================================================
        # Embedded RK4(5) IP Method - FiberNlse algorithm
        # ============================================================

        # Linear half-step operator: exp(D̂·dz/2)
        @. e = exp(0.5 * dz * model.dispersion_term)

        # Transform to interaction picture: Ûᵢₚ = exp(D̂z/2)·U
        @. Uip = e * U

        # Stage 1: k₁ = exp(D̂z/2)·N̂[A(z)]
        @. k1 = e * NU

        # Stage 2: k₂ = N̂[IFFT(Ûᵢₚ + dz/2·k₁)]
        @. Aw_temp = Uip + 0.5 * dz * k1
        mul!(At_temp, model.ifftp, Aw_temp)  # Convert to time domain
        k2 .= model.nonlinear_function(At_temp, model)

        # Stage 3: k₃ = N̂[IFFT(Ûᵢₚ + dz/2·k₂)]
        @. Aw_temp = Uip + 0.5 * dz * k2
        mul!(At_temp, model.ifftp, Aw_temp)
        k3 .= model.nonlinear_function(At_temp, model)

        # Stage 4: k₄ = N̂[IFFT(exp(D̂z/2)·(Ûᵢₚ + dz·k₃))]
        @. Aw_temp = e * (Uip + dz * k3)
        mul!(At_temp, model.ifftp, Aw_temp)
        k4 .= model.nonlinear_function(At_temp, model)

        # Accumulator: r = exp(D̂z/2)·(Ûᵢₚ + dz·(k₁/6 + k₂/3 + k₃/3))
        @. r = e * (Uip + dz * (k1 / 6.0 + k2 / 3.0 + k3 / 3.0))

        # 4th order solution: U⁽⁴⁾ = r + dz * k4 / 6.0

        # Stage 5: k₅ = N̂[IFFT(U⁽⁴⁾)]
        mul!(At_temp, model.ifftp, r + dz * k4 / 6.0)
        k5 .= model.nonlinear_function(At_temp, model)

        # 5th order solution: U2 = r + dz·(k₄/15 + k₅/10)
        @. U2 = r + dz * (k4 / 15.0 + k5 / 10.0)
        # 4th order solution: U1 = r + dz·k₄/6
        @. U1 = r + dz * k4 / 6.0

        # ============================================================
        # Error Estimation & Adaptive Stepping
        # ============================================================

        # Compute local error: ‖U⁽⁵⁾ - U⁽⁴⁾‖ / ‖U⁽⁴⁾‖
        local_error = sqrt(sum(abs2, U2 .- U1) / sum(abs2, U1))

        # Optimal step size (FiberNlse formula)
        dzopt = max(0.5, min(2.0, 0.9 * sqrt(sqrt(rtol / local_error)))) * dz

        if local_error <= rtol
            # Accept step
            dz = min(dzopt, abs(z_target - z))
            z += dz
            copyto!(U, U1)        # Use 4th-order solution (FiberNlse choice)
            copyto!(NU, k5)       # FSAL: reuse k₅ as next N̂[A]

            # Save output if we reached target z
            if z >= z_target - 1e-12 * z_end
                z_out[save_idx] = z
                copyto!(@view(Aw_out[:, save_idx]), U)
                # Transform to time domain and shift to natural order
                mul!(At_temp, model.ifftp, U)
                At_out[:, save_idx] .= fftshift(At_temp)

                # Update progress bar
                if !isnothing(prog)
                    update!(prog, save_idx - 1)
                end

                save_idx += 1
            end
        else
            # Reject step - reduce dz and retry
            dz = dzopt
        end
    end

    return z_out, At_out, Aw_out
end
