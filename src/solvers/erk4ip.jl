"""
Embedded Runge-Kutta 4(3) in Interaction Picture (ERK4IP) solver with adaptive step-size.

This implementation uses RK4 for the main integration and RK3 for error estimation,
allowing automatic step-size adaptation for efficient and accurate simulation.

Based on the interaction picture transformation and Bogacki-Shampine coefficients.
"""

"""
    propagate_erk4ip(pulse::Pulse, params::SimParams; 
                     progress::Bool=true, rtol::Float64=1e-6, atol::Float64=1e-8)

Propagate pulse using Embedded Runge-Kutta 4(3) in Interaction Picture with adaptive stepping.

This method adaptively adjusts the step size based on local error estimates, providing
efficient integration while maintaining accuracy. Uses RK4 for the integration and RK3
for error estimation.

# Arguments
- `pulse::Pulse`: Initial pulse
- `params::SimParams`: Simulation parameters
- `progress::Bool`: Show progress bar (default: true)
- `rtol::Float64`: Relative tolerance for adaptive stepping (default: 1e-6)
- `atol::Float64`: Absolute tolerance for adaptive stepping (default: 1e-8)

# Returns
- `Tuple{Vector{Float64}, Matrix{ComplexF64}, Matrix{ComplexF64}}`: (z, At, Aw)

# Notes
- Automatically handles frequency-dependent gamma using M-GNLSE pseudo-envelope method
- Adaptive step sizing ensures accuracy while minimizing computational cost
- Butcher tableau uses Bogacki-Shampine coefficients (FSAL property)

# References
P. Bogacki and L.F. Shampine, "A 3(2) pair of Runge-Kutta formulas,"  
Appl. Math. Lett. 2(4), 321-325 (1989).
"""
function propagate_erk4ip(pulse::Pulse, params::SimParams; 
                          progress::Bool=true, 
                          rtol::Float64=1e-6, 
                          atol::Float64=1e-8)
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
    
    # Pre-allocate buffers
    exp_Lz_buffer = similar(pulse.Aw)
    exp_mLz_buffer = similar(pulse.Aw)
    
    # Butcher tableau for Bogacki-Shampine RK4(3) method
    # This has the First Same As Last (FSAL) property for efficiency
    a21 = 1/2
    a32 = 0.0
    a31 = 3/4
    a43 = 2/9
    a42 = 1/3
    a41 = 2/9
    
    c2 = 1/2
    c3 = 3/4
    c4 = 1.0
    
    # 4th order weights
    b1 = 2/9
    b2 = 1/3
    b3 = 4/9
    b4 = 0.0  # FSAL: k4 becomes k1 of next step
    
    # 3rd order weights (for error estimation)
    b1_hat = 7/24
    b2_hat = 1/4
    b3_hat = 1/3
    b4_hat = 1/8
    
    # Initial condition and parameter setup
    local u0, rhs_func
    
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
        
        # Pre-allocate nonlinearity buffers
        At_buffer = similar(pulse.At)
        nonlin_w = similar(pulse.Aw)
        
        if scaling !== nothing
            u0 = copy(pulse.Aw) .* scaling
            inv_scaling = 1.0 ./ scaling
            Aw_buffer = similar(pulse.Aw)
            
            # Create RHS function closure using proper nonlinear operator
            rhs_func = (u, z) -> begin
                du = similar(u)
                @. exp_Lz_buffer = exp(linop * z)
                @. exp_mLz_buffer = exp(-linop * z)
                @. du = u * exp_Lz_buffer
                @. Aw_buffer = du * inv_scaling
                mul!(At_buffer, fft_plan, Aw_buffer)
                # nonlinear_operator_frequency_dependent returns frequency-domain nonlinearity
                nonlin_w .= nonlinear_operator_frequency_dependent(
                    At_buffer, gamma_im_vec, omega0_inv, fr, one_minus_fr,
                    omega, fft_plan, ifft_plan, RW, raman, shock, grid.dt
                )
                @. nonlin_w *= scaling
                @. du = nonlin_w * exp_mLz_buffer
                return du
            end
        else
            u0 = copy(pulse.Aw)
            
            # Create RHS function closure using proper nonlinear operator
            rhs_func = (u, z) -> begin
                du = similar(u)
                @. exp_Lz_buffer = exp(linop * z)
                @. exp_mLz_buffer = exp(-linop * z)
                @. du = u * exp_Lz_buffer
                mul!(At_buffer, fft_plan, du)
                # nonlinear_operator_frequency_dependent returns frequency-domain nonlinearity
                nonlin_w .= nonlinear_operator_frequency_dependent(
                    At_buffer, gamma_im_vec, omega0_inv, fr, one_minus_fr,
                    omega, fft_plan, ifft_plan, RW, raman, shock, grid.dt
                )
                @. du = nonlin_w * exp_mLz_buffer
                # Debug: check for NaN/Inf
                if any(isnan, du) || any(isinf, du)
                    error("NaN or Inf in RHS at z=$z")
                end
                return du
            end
        end
    else
        # Scalar gamma
        u0 = copy(pulse.Aw)
        
        # Create RHS function closure
        rhs_func = (u, z) -> begin
            du = similar(u)
            @. exp_Lz_buffer = exp(linop * z)
            @. exp_mLz_buffer = exp(-linop * z)
            @. du = u * exp_Lz_buffer
            At = fft_plan * du
            # Calculate nonlinear RHS: N[A] = iγ|A|²·A
            # nonlinear_operator returns iγ|A|², so multiply by A
            nonlin_phase = nonlinear_operator(At, params, grid, RW)
            nonlin = @. At * nonlin_phase
            du .= ifft_plan * nonlin
            @. du *= exp_mLz_buffer
            return du
        end
    end
    
    # Relative tolerance (following FiberNlse.jl new_version approach)
    reltol = params.reltol
    
    # Adaptive stepping parameters
    z = 0.0
    z_end = medium.length
    h = medium.length / 100  # Initial step size
    h_min = medium.length / 1e6  # Minimum step size to prevent infinite reduction
    h_max = medium.length / (params.n_saves - 1)  # Maximum step size (distance between saves)
    maxiter = 100000  # Maximum number of iterations to prevent infinite loops
    
    # Storage for solution at requested save points
    z_saves = range(0, z_end, length=params.n_saves)
    save_idx = 1
    z_out = zeros(params.n_saves)
    u_out = [zeros(ComplexF64, N) for _ in 1:params.n_saves]
    z_out[1] = 0.0
    u_out[1] .= u0
    save_idx = 2
    
    u = copy(u0)
    
    if progress
        println("Starting ERK4IP integration with adaptive stepping...")
        println("  Fiber length: $(z_end*1000) mm")
        println("  Initial step size: $(h*1e6) μm")
        println("  Save points: $(params.n_saves)")
        println("  Relative tolerance: reltol=$reltol")
    end
    
    # First RHS evaluation
    k1 = rhs_func(u, z)
    
    # Adaptive stepping loop
    step_count = 0
    rejected_steps = 0
    
    while z < z_end && step_count < maxiter
        # Check if we've collected all save points
        if save_idx > params.n_saves
            if progress
                println("All save points collected, exiting integration loop.")
            end
            break
        end
        
        step_count += 1
        
        # Ensure we don't overshoot
        if z + h > z_end
            h = z_end - z
        end
        
        # RK stages
        k2 = rhs_func(u .+ h * a21 * k1, z + c2 * h)
        k3 = rhs_func(u .+ h * (a31 * k1 .+ a32 * k2), z + c3 * h)
        k4 = rhs_func(u .+ h * (a41 * k1 .+ a42 * k2 .+ a43 * k3), z + c4 * h)
        
        # 4th order solution
        u_new = u .+ h * (b1 * k1 .+ b2 * k2 .+ b3 * k3 .+ b4 * k4)
        
        # 3rd order solution (for error estimation)
        u_hat = u .+ h * (b1_hat * k1 .+ b2_hat * k2 .+ b3_hat * k3 .+ b4_hat * k4)
        
        # Error estimate (normalized RMS norm - matches FiberNlse.jl exactly)
        error = u_new .- u_hat
        error_norm = sqrt(sum(abs2, error) / sum(abs2, u_new))
        
        # Debug: print error norm occasionally
        if step_count % 1000 == 1 && progress
            println("  Debug: step $step_count, error_norm=$error_norm, h=$(h*1e6) μm")
        end
        
        # Compute optimal step size (matches FiberNlse.jl exactly)
        dzopt = max(0.5, min(2.0, 0.9 * sqrt(sqrt(reltol / max(error_norm, 1e-14))))) * h
        
        # Check if step is accepted
        if error_norm <= reltol || h <= h_min
            # Accept step - advance with CURRENT step size (the one that produced u_new)
            h_accepted = h
            z += h_accepted
            u .= u_new
            
            # NOW compute the next step size for the NEXT iteration
            h = min(dzopt, abs(z_end - z))
            
            # Save solution at requested save points (with interpolation if needed)
            while save_idx <= params.n_saves && z >= z_saves[save_idx]
                z_save = z_saves[save_idx]
                z_out[save_idx] = z_save
                u_out[save_idx] .= u
                save_idx += 1
            end
            
            # If we've reached the end, save any remaining save points
            if z >= z_end - 1e-10 * z_end
                while save_idx <= params.n_saves
                    z_out[save_idx] = z_end
                    u_out[save_idx] .= u
                    save_idx += 1
                end
            end
            
            if progress && (step_count % 100 == 0 || z >= z_end)
                percent = 100 * z / z_end
                println("  Progress: $(round(percent, digits=1))% | z=$(round(z*1000, digits=2)) mm | h=$(round(h*1e6, digits=2)) μm | steps=$step_count | saved=$(save_idx-1)/$(params.n_saves)")
            end
            
            # FSAL property: reuse last evaluation for next step
            k1 = k4
        else
            # Reject step and retry with smaller step size
            rejected_steps += 1
            h = dzopt
            h = clamp(h, h_min, h_max)
        end
    end
    
    # Check if we hit maxiter
    if step_count >= maxiter && z < z_end
        @warn "ERK4IP reached maximum iterations ($maxiter) before completing integration. Consider increasing maxiter or relaxing tolerances."
    end
    
    if progress
        println("ERK4IP completed: $step_count steps ($rejected_steps rejected)")
    end
    
    # Transform back from interaction picture
    At_out = zeros(ComplexF64, N, params.n_saves)
    Aw_out = zeros(ComplexF64, N, params.n_saves)
    
    # Special handling for z=0: directly copy initial condition without transformation
    At_out[:, 1] = copy(pulse.At)
    Aw_out[:, 1] = copy(pulse.Aw)
    
    if is_freq_gamma && scaling !== nothing
        inv_scaling = 1.0 ./ scaling
        for i in 2:params.n_saves
            z_val = z_out[i]
            u_val = u_out[i]
            @. exp_Lz_buffer = exp(linop * z_val)
            Aw_pseudo = @. u_val * exp_Lz_buffer
            @. Aw_out[:, i] = Aw_pseudo * inv_scaling
            At_out[:, i] = fft_plan * Aw_out[:, i]
        end
    elseif is_freq_gamma
        for i in 2:params.n_saves
            z_val = z_out[i]
            u_val = u_out[i]
            @. exp_Lz_buffer = exp(linop * z_val)
            @. Aw_out[:, i] = u_val * exp_Lz_buffer
            At_out[:, i] = fft_plan * Aw_out[:, i]
        end
    else
        for i in 2:params.n_saves
            z_val = z_out[i]
            u_val = u_out[i]
            @. exp_Lz_buffer = exp(linop * z_val)
            @. Aw_out[:, i] = u_val * exp_Lz_buffer
            At_out[:, i] = fft_plan * Aw_out[:, i]
        end
    end
    
    (z_out, At_out, Aw_out)
end
