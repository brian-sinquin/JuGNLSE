"""
    gnlse_rhs!(du, u, p, z)

Right-hand side of GNLSE in the interaction picture for ODE solver (scalar gamma).
Optimized - no conditionals in hot path.

# Arguments
- `du`: Output derivative
- `u`: Current state (interaction picture)
- `p`: Parameters tuple (linop, params, grid, RW, fft_plan, ifft_plan, exp_Lz_buffer, exp_mLz_buffer)
- `z`: Current propagation distance [m]
"""
function gnlse_rhs!(du, u, p, z)
    linop, params, grid, RW, fft_plan, ifft_plan, exp_Lz_buffer, exp_mLz_buffer = p
    
    # Pre-compute exponentials
    @. exp_Lz_buffer = exp(linop * z)
    @. exp_mLz_buffer = exp(-linop * z)
    
    # Transform back from interaction picture: A_w = u * exp(L*z)
    @. du = u * exp_Lz_buffer
    
    # Transform to time domain
    At = fft_plan * du
    
    # Calculate nonlinear RHS: N[A] = iγ|A|²·A
    # nonlinear_operator returns iγ|A|², so multiply by A
    nonlin_phase = nonlinear_operator(At, params, grid, RW)
    nonlin = @. At * nonlin_phase
    
    # Transform back to frequency domain
    du .= ifft_plan * nonlin
    
    # Apply interaction picture factor
    @. du *= exp_mLz_buffer
    
    nothing
end

"""
    gnlse_rhs_freq_gamma!(du, u, p, z)

Right-hand side of M-GNLSE with frequency-dependent gamma (with scaling).
Optimized - no conditionals in hot path.

# Arguments
- `du`: Output derivative
- `u`: Current state (pseudo-envelope in interaction picture)
- `p`: Parameters tuple (linop, params, grid, RW, fft_plan, ifft_plan, scaling, 
                         inv_scaling, exp_Lz_buffer, exp_mLz_buffer, Aw_buffer)
- `z`: Current propagation distance [m]
"""
function gnlse_rhs_freq_gamma!(du, u, p, z)
    linop, gamma_im_vec, omega0_inv, fr, one_minus_fr, omega, fft_plan, ifft_plan, RW, raman, shock,
        scaling, inv_scaling, exp_Lz_buffer, exp_mLz_buffer, Aw_buffer, At_buffer, It_buffer, nonlin_w = p
    
    # Pre-compute exponentials
    @. exp_Lz_buffer = exp(linop * z)
    @. exp_mLz_buffer = exp(-linop * z)
    
    # Transform from interaction picture: Aw_pseudo = u * exp(L*z)
    @. du = u * exp_Lz_buffer
    
    # Remove scaling: Aw = Aw_pseudo / scaling
    @. Aw_buffer = du * inv_scaling
    
    # Transform to time domain (in-place)
    mul!(At_buffer, fft_plan, Aw_buffer)
    
    # Calculate nonlinear operator inline (Kerr-only, optimized)
    @. It_buffer = abs2(At_buffer)
    @. At_buffer *= It_buffer  # Reuse At_buffer for nonlin_t
    
    # Transform to frequency domain (in-place)
    mul!(nonlin_w, ifft_plan, At_buffer)
    @. nonlin_w *= gamma_im_vec
    
    # Apply scaling: result is for pseudo-envelope
    @. nonlin_w *= scaling
    
    # Apply interaction picture factor
    @. du = nonlin_w * exp_mLz_buffer
    
    nothing
end

"""
    gnlse_rhs_freq_gamma_no_scaling!(du, u, p, z)

Right-hand side of M-GNLSE with frequency-dependent gamma (without scaling).
Optimized - no conditionals in hot path.
"""
function gnlse_rhs_freq_gamma_no_scaling!(du, u, p, z)
    linop, gamma_im_vec, omega0_inv, fr, one_minus_fr, omega, fft_plan, ifft_plan, RW, raman, shock,
        exp_Lz_buffer, exp_mLz_buffer, At_buffer, It_buffer, nonlin_w = p
    
    # Pre-compute exponentials
    @. exp_Lz_buffer = exp(linop * z)
    @. exp_mLz_buffer = exp(-linop * z)
    
    # Transform from interaction picture
    @. du = u * exp_Lz_buffer
    
    # Transform to time domain (in-place)
    mul!(At_buffer, fft_plan, du)
    
    # Calculate nonlinear operator inline (Kerr-only, optimized)
    @. It_buffer = abs2(At_buffer)
    @. At_buffer *= It_buffer  # Reuse At_buffer for nonlin_t
    
    # Transform to frequency domain (in-place)
    mul!(nonlin_w, ifft_plan, At_buffer)
    @. nonlin_w *= gamma_im_vec
    
    # Apply interaction picture factor
    @. du = nonlin_w * exp_mLz_buffer
    
    nothing
end

"""
    propagate_rk4ip(pulse::Pulse, params::SimParams; progress::Bool=true)

Propagate pulse using RK4 in the Interaction Picture (RK4IP) with adaptive stepping.

Uses OrdinaryDiffEq.jl for efficient integration with error control.
Automatically detects and handles frequency-dependent gamma using M-GNLSE pseudo-envelope method.
Optimized with pre-allocated buffers and no conditionals in hot path.

# Arguments
- `pulse::Pulse`: Initial pulse
- `params::SimParams`: Simulation parameters
- `progress::Bool`: Show progress bar (default: true)

# Returns
- `Tuple{Vector{Float64}, Matrix{ComplexF64}, Matrix{ComplexF64}}`: (z, At, Aw)

# Notes
For frequency-dependent gamma, the solver applies the Lægsgaard (2007) pseudo-envelope
transformation if a scaling factor is provided in the Medium structure.
"""
function propagate_rk4ip(pulse::Pulse, params::SimParams; progress::Bool=true)
    # Setup
    grid = pulse.grid
    medium = params.medium
    N = grid.N
    
    # Dispersion operator
    linop = dispersion_operator(grid, medium)
    
    # Raman response (if needed)
    RW = nothing
    if params.raman
        h_R, _ = raman_response(grid, params.raman_model)
        RW = raman_response_frequency(h_R, grid)
    end
    
    # FFT plans
    fft_plan = plan_fft(pulse.At)
    ifft_plan = plan_ifft(pulse.Aw)
    
    # Determine if using frequency-dependent gamma
    is_freq_gamma = medium.gamma isa Vector
    scaling = medium.scaling
    
    # Pre-allocate buffers (avoid allocations in hot path)
    exp_Lz_buffer = similar(pulse.Aw)
    exp_mLz_buffer = similar(pulse.Aw)
    
    # Initial condition and RHS function selection
    local u0, p, rhs!
    
    if is_freq_gamma
        # Pre-compute constants for frequency-dependent gamma
        gamma_im_vec = im .* medium.gamma
        omega0 = 2π * 3e8 / medium.lambda0
        omega0_inv = 1.0 / omega0
        fr = params.fr
        one_minus_fr = 1.0 - fr
        omega = grid.omega
        raman = params.raman
        shock = params.shock
        
        # Pre-allocate buffers for nonlinearity computation
        At_buffer = similar(pulse.At)           # Time-domain field
        It_buffer = similar(pulse.At, Float64)  # Real intensity
        nonlin_w = similar(pulse.Aw)            # Frequency-domain nonlinear term
        
        if scaling !== nothing
            # With scaling - fully optimized path
            u0 = copy(pulse.Aw) .* scaling
            inv_scaling = 1.0 ./ scaling  # Pre-compute inverse
            Aw_buffer = similar(pulse.Aw)
            p = (linop, gamma_im_vec, omega0_inv, fr, one_minus_fr, omega, fft_plan, ifft_plan, 
                 RW, raman, shock, scaling, inv_scaling, exp_Lz_buffer, exp_mLz_buffer, Aw_buffer,
                 At_buffer, It_buffer, nonlin_w)
            rhs! = gnlse_rhs_freq_gamma!
        else
            # Without scaling
            u0 = copy(pulse.Aw)
            p = (linop, gamma_im_vec, omega0_inv, fr, one_minus_fr, omega, fft_plan, ifft_plan,
                 RW, raman, shock, exp_Lz_buffer, exp_mLz_buffer, At_buffer, It_buffer, nonlin_w)
            rhs! = gnlse_rhs_freq_gamma_no_scaling!
        end
    else
        # Scalar gamma
        u0 = copy(pulse.Aw)
        p = (linop, params, grid, RW, fft_plan, ifft_plan, exp_Lz_buffer, exp_mLz_buffer)
        rhs! = gnlse_rhs!
    end
    
    tspan = (0.0, medium.length)
    
    # Solve ODE
    prob = ODEProblem(rhs!, u0, tspan, p)
    sol = OrdinaryDiffEq.solve(prob, DP5(), 
                reltol=params.reltol, 
                abstol=params.abstol,
                saveat=range(0, medium.length, length=params.n_saves),
                progress=progress)
    
    # Extract solution
    z_array = sol.t
    n_saves = length(z_array)
    
    # Transform back from interaction picture and organize output
    At_out = zeros(ComplexF64, N, n_saves)
    Aw_out = zeros(ComplexF64, N, n_saves)
    
    # Post-processing based on solver type
    if is_freq_gamma && scaling !== nothing
        inv_scaling = 1.0 ./ scaling
        for i in 1:n_saves
            z = z_array[i]
            u = sol.u[i]
            @. exp_Lz_buffer = exp(linop * z)
            Aw_pseudo = @. u * exp_Lz_buffer
            @. Aw_out[:, i] = Aw_pseudo * inv_scaling
            At_out[:, i] = fft_plan * Aw_out[:, i]
        end
    elseif is_freq_gamma
        for i in 1:n_saves
            z = z_array[i]
            u = sol.u[i]
            @. exp_Lz_buffer = exp(linop * z)
            @. Aw_out[:, i] = u * exp_Lz_buffer
            At_out[:, i] = fft_plan * Aw_out[:, i]
        end
    else
        for i in 1:n_saves
            z = z_array[i]
            u = sol.u[i]
            @. exp_Lz_buffer = exp(linop * z)
            @. Aw_out[:, i] = u * exp_Lz_buffer
            At_out[:, i] = fft_plan * Aw_out[:, i]
        end
    end
    
    (z_array, At_out, Aw_out)
end
