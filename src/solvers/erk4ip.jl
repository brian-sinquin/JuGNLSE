

import ..build_physics_model, ..PhysicsModel
using ..JuGNLSE: GNLSEProblem, Solution, Pulse, SimParams, AbstractGammaCoefficient, photon_number

"""
    ERK4IP <: AbstractGNLSESolver

An adaptive embedded Runge-Kutta 4th-order in the interaction picture (ERK4IP) solver for the GNLSE.
This solver dynamically adjusts the propagation step size to maintain a specified error tolerance,
making it efficient for problems with varying nonlinearity or dispersion.

# Fields
  - `rtol::Float64`: Relative tolerance for adaptive step size control (default: 1e-6).
  - `atol::Float64`: Absolute tolerance for adaptive step size control (default: 1e-8).
  - `dz::Union{Float64, Nothing}`: Initial propagation step size [m]. If `nothing` (default),
    an initial step size is determined automatically (e.g., `medium.length / 1000`).

# Constructors
  - `ERK4IP(; rtol=1e-6, atol=1e-8, dz=nothing)`: Keyword constructor.
"""
struct ERK4IP <: AbstractGNLSESolver
    rtol::Float64
    atol::Float64
    dz::Union{Float64, Nothing}
end

ERK4IP(; rtol::Float64=1e-6, atol::Float64=1e-8, dz::Union{Float64, Nothing}=nothing) = ERK4IP(rtol, atol, dz)

"""
    solve(problem::GNLSEProblem, solver::ERK4IP; progress::Bool=true)

Solves the GNLSE using the `ERK4IP` adaptive embedded Runge-Kutta solver.

# Arguments
  - `problem::GNLSEProblem`: The GNLSE problem definition.
  - `solver::ERK4IP`: The `ERK4IP` solver instance, containing tolerances and optional initial step size.
  - `progress::Bool=true`: If `true`, a progress bar will be displayed during propagation.

# Returns
  - `Solution`: A `Solution` object containing the pulse's evolution through the fiber.

# Notes
This method orchestrates the GNLSE solution by building the `PhysicsModel` and then calling the internal `_propagate_erk4ip!` function.
It also performs a photon number conservation check for lossless fibers and issues a warning if significant drift is detected.
"""
function solve(problem::GNLSEProblem, solver::ERK4IP; progress::Bool=true)
    pulse = problem.initial_pulse
    params = problem.sim_params
    gamma_coefficient = problem.gamma_coefficient

    model = build_physics_model(pulse.grid, params, gamma_coefficient)
    z, At, AW = _propagate_erk4ip!(model, pulse, params, progress, solver.rtol, solver.atol, solver.dz)

    # Build solution
    grid = pulse.grid
    solution = Solution(
        grid.t,          # Time grid [s]
        grid.W,          # Absolute frequency [rad/s]
        grid.omega0,     # Central frequency [rad/s]
        z,               # Propagation distances [m]
        At,              # Time domain fields (N × z_saves)
        AW,              # Frequency domain fields (N × z_saves)
    )

    if params.medium.loss == 0
        n = photon_number(solution)
        drift = abs(n[end] - n[1]) / n[1]
        drift > 1e-2 && @warn "Photon number drifted by " *
            "$(round(100 * drift; digits=2))% — consider a tighter `rtol`/`atol`."
    end

    return solution
end

"""
    _propagate_erk4ip!(model::PhysicsModel, pulse::Pulse, params::SimParams, progress::Bool, rtol::Float64, atol::Float64, dz_init::Union{Float64, Nothing})

Internal function to propagate an optical pulse using the ERK4IP adaptive solver.
This function performs the core numerical integration of the GNLSE.

# Arguments
  - `model::PhysicsModel`: The pre-computed physics model containing operators and FFT plans.
  - `pulse::Pulse`: The optical pulse to propagate. Its frequency domain representation `AW` will be modified in-place during propagation.
  - `params::SimParams`: Simulation parameters, including fiber length, number of save points, and physics flags.
  - `progress::Bool`: If `true`, a progress bar will be displayed.
  - `rtol::Float64`: Relative tolerance for adaptive step size control.
  - `atol::Float64`: Absolute tolerance for adaptive step size control.
  - `dz_init::Union{Float64, Nothing}`: Initial propagation step size [m]. If `nothing`, an initial step size is determined automatically.

# Returns
  - `z_out::Vector{Float64}`: Vector of propagation distances [m] at which the pulse state was saved.
  - `At_out::Matrix{ComplexF64}`: Matrix of time-domain pulse envelopes (N × n_saves) at each saved distance.
  - `Aw_out::Matrix{ComplexF64}`: Matrix of frequency-domain pulse envelopes (N × n_saves) at each saved distance.

# Notes
This function is an internal implementation detail of the `ERK4IP` solver and is not intended for direct external use.
It handles the adaptive step sizing, Runge-Kutta stages, and saving of intermediate pulse states.
"""
function _propagate_erk4ip!(
    model::PhysicsModel,
    pulse::Pulse,
    params::SimParams,
    progress::Bool,
    rtol::Float64,
    atol::Float64,
    dz_init::Union{Float64, Nothing},
)
    grid = pulse.grid
    N = grid.N
    n_saves = params.z_saves
    z_end::Float64 = params.medium.length

    U = copy(pulse.AW)

    z_out = zeros(n_saves)
    At_out = zeros(ComplexF64, N, n_saves)
    Aw_out = zeros(ComplexF64, N, n_saves)

    z_out[1] = 0.0
    At_out[:, 1] .= pulse.At
    Aw_out[:, 1] .= fftshift(U)

    z = 0.0
    dz::Float64 = dz_init === nothing ? z_end / 1000 : dz_init
    save_idx = 2
    z_saves = range(0, z_end; length=n_saves)

    k1 = similar(U)
    k2 = similar(U)
    k3 = similar(U)
    k4 = similar(U)
    k5 = similar(U)
    Nu = similar(U)
    U_temp = similar(U)
    u_temp = similar(pulse.At)
    exp_half_dz_D = similar(U)

    r = similar(U)
    U4_fft = similar(U)
    U5_fft = similar(U)
    u4 = similar(pulse.At)

    prog = progress ? Progress(n_saves - 1; desc="ERK4IP: ", showspeed=true) : nothing

    step_count = 0
    rejected_steps = 0

    mul!(u_temp, model.to_time, U)
    copyto!(Nu, model.nonlinear_function(u_temp, model, z))

    while z < z_end && save_idx <= n_saves
        z_target = z_saves[save_idx]
        dz = min(dz, z_target - z)

        @. exp_half_dz_D = exp(0.5 * dz * model.D)

        @. k1 = exp_half_dz_D * Nu

        @. U_temp = exp_half_dz_D * U + 0.5 * dz * k1
        mul!(u_temp, model.to_time, U_temp)
        copyto!(k2, model.nonlinear_function(u_temp, model, z + 0.5 * dz))

        @. U_temp = exp_half_dz_D * U + 0.5 * dz * k2
        mul!(u_temp, model.to_time, U_temp)
        copyto!(k3, model.nonlinear_function(u_temp, model, z + 0.5 * dz))

        @. U_temp = exp_half_dz_D * (exp_half_dz_D * U + dz * k3)
        mul!(u_temp, model.to_time, U_temp)
        copyto!(k4, model.nonlinear_function(u_temp, model, z + dz))

        @. r = exp_half_dz_D * (exp_half_dz_D * U + dz * (k1 / 6.0 + k2 / 3.0 + k3 / 3.0))

        @. U4_fft = r + (dz / 6.0) * k4

        mul!(u4, model.to_time, U4_fft)
        copyto!(k5, model.nonlinear_function(u4, model, z + dz))

        @. U5_fft = r + (dz / 15.0) * k4 + (dz / 10.0) * k5

        err2 = zero(Float64)
        @simd for i in eachindex(U4_fft, U5_fft)
            sc = atol + rtol * abs(U4_fft[i])
            err2 += abs2(U4_fft[i] - U5_fft[i]) / (sc * sc)
        end
        local_error = sqrt(err2 / N)

        safety = 0.9
        exponent = 0.25

        if local_error <= 1.0
            step_count += 1
            z += dz
            copyto!(U, U4_fft)

            copyto!(Nu, k5)

            factor = safety * (1.0 / (local_error + 1e-300))^exponent
            factor = max(0.5, min(2.0, factor))
            dz = factor * dz

            if z >= z_target - 1e-12 * z_end
                z_out[save_idx] = z

                copyto!(model.buf_f1, U)

                fftshift!(@view(Aw_out[:, save_idx]), model.buf_f1)

                mul!(u_temp, model.to_time, model.buf_f1)
                copyto!(@view(At_out[:, save_idx]), u_temp)

                if !isnothing(prog)
                    update!(prog, save_idx - 1)
                end

                save_idx = 2
            end
        else
            rejected_steps += 1
            factor = safety * (1.0 / (local_error + 1e-300))^exponent
            factor = max(0.5, min(2.0, factor))
            dz = factor * dz
        end
    end

    if progress && !isnothing(prog)
        println("\n✓ Steps: $step_count accepted, $rejected_steps rejected")
    end

    return z_out, At_out, Aw_out
end
