

struct RK4 <: AbstractGNLSESolver
    dz::Float64
end

RK4(; dz::Float64=1e-3) = RK4(dz)

function solve(problem::GNLSEProblem, solver::RK4; progress::Bool=true)
    pulse = problem.initial_pulse
    params = problem.sim_params
    gamma_coefficient = problem.gamma_coefficient

    model = build_physics_model(pulse.grid, params, gamma_coefficient)
    z, At, AW = _propagate_rk4!(model, pulse, params, progress, solver.dz)

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
            "$(round(100 * drift; digits=2))% — consider a tighter `dz`."
    end

    return solution
end

function _propagate_rk4!(
    model::PhysicsModel,
    pulse::Pulse,
    params::SimParams,
    progress::Bool,
    dz::Float64,
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
    save_idx = 2
    z_saves = range(0, z_end; length=n_saves)

    # Pre-allocate workspace
    k1_f = similar(U) # frequency domain
    k2_f = similar(U)
    k3_f = similar(U)
    k4_f = similar(U)
    u_temp_f = similar(U) # frequency domain temp
    u_temp_t = similar(pulse.At) # time domain temp

    prog = progress ? Progress(n_saves - 1; desc="RK4: ", showspeed=true) : nothing

    while z < z_end && save_idx <= n_saves
        # Determine actual step size (ensure we don't overshoot z_end or z_target)
        actual_dz = min(dz, z_end - z, z_saves[save_idx] - z)

        # RK4 stages
        # k1
        mul!(u_temp_t, model.to_time, U) # U (freq) -> u (time)
        copyto!(k1_f, model.nonlinear_function(u_temp_t, model, z)) # N(u) (time) -> N(u) (freq)
        @. k1_f = model.D * U + k1_f # D*U + N(u)

        # k2
        @. u_temp_f = U + 0.5 * actual_dz * k1_f # U + 0.5*h*k1
        mul!(u_temp_t, model.to_time, u_temp_f) # u_temp (freq) -> u_temp (time)
        copyto!(k2_f, model.nonlinear_function(u_temp_t, model, z + 0.5 * actual_dz))
        @. k2_f = model.D * u_temp_f + k2_f

        # k3
        @. u_temp_f = U + 0.5 * actual_dz * k2_f
        mul!(u_temp_t, model.to_time, u_temp_f)
        copyto!(k3_f, model.nonlinear_function(u_temp_t, model, z + 0.5 * actual_dz))
        @. k3_f = model.D * u_temp_f + k3_f

        # k4
        @. u_temp_f = U + actual_dz * k3_f
        mul!(u_temp_t, model.to_time, u_temp_f)
        copyto!(k4_f, model.nonlinear_function(u_temp_t, model, z + actual_dz))
        @. k4_f = model.D * u_temp_f + k4_f

        # Update U
        @. U = U + (actual_dz / 6.0) * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)
        z += actual_dz

        # Save output at target distance if crossed
        if z >= z_saves[save_idx] - 1e-12 * z_end # Use tolerance for float comparison
            z_out[save_idx] = z

            copyto!(model.buf_f1, U)

            fftshift!(@view(Aw_out[:, save_idx]), model.buf_f1)

            mul!(u_temp_t, model.to_time, model.buf_f1)
            copyto!(@view(At_out[:, save_idx]), u_temp_t)

            if !isnothing(prog)
                update!(prog, save_idx - 1)
            end

            save_idx += 1
        end
    end

    return z_out, At_out, Aw_out
end
