"""
    propagate_ssfm(pulse::Pulse, params::SimParams; progress::Bool=true)

Propagate pulse using Split-Step Fourier Method (SSFM).

Simple second-order accurate method (Strang splitting).
Automatically detects and handles frequency-dependent gamma using M-GNLSE pseudo-envelope method.

# Arguments
- `pulse::Pulse`: Initial pulse
- `params::SimParams`: Simulation parameters
- `progress::Bool`: Show progress bar (default: true)

# Returns
- `Tuple{Vector{Float64}, Matrix{ComplexF64}, Matrix{ComplexF64}}`: (z, At, Aw)
  where z is propagation distances, At is time-domain evolution, Aw is frequency-domain evolution

# Notes
For frequency-dependent gamma, the solver applies the Lægsgaard (2007) pseudo-envelope
transformation if a scaling factor is provided in the Medium structure.
"""
function propagate_ssfm(pulse::Pulse, params::SimParams; progress::Bool=true)
    # Setup
    grid = pulse.grid
    medium = params.medium
    N = grid.N
    n_saves = params.n_saves
    
    # Propagation distances
    z_array = range(0, medium.length, length=n_saves)
    dz = medium.length / (n_saves - 1)
    
    # Initialize output arrays
    At_out = zeros(ComplexF64, N, n_saves)
    Aw_out = zeros(ComplexF64, N, n_saves)
    
    # Initial condition
    At = copy(pulse.At)
    Aw = copy(pulse.Aw)
    
    # Determine if using frequency-dependent gamma
    is_freq_gamma = medium.gamma isa Vector
    scaling = medium.scaling
    
    # Apply pseudo-envelope scaling for initial condition if needed
    if is_freq_gamma && scaling !== nothing
        Aw .*= scaling
    end
    
    At_out[:, 1] = copy(pulse.At)  # Store unscaled
    Aw_out[:, 1] = copy(pulse.Aw)  # Store unscaled
    
    # Dispersion operator
    linop = dispersion_operator(grid, medium)
    
    # Raman response (if needed)
    RW = nothing
    if params.raman
        h_R, _ = raman_response(grid, params.raman_model)
        RW = raman_response_frequency(h_R, grid)
    end
    
    # FFT plans
    fft_plan = plan_fft(At)
    ifft_plan = plan_ifft(Aw)
    
    # Pre-compute constants for frequency-dependent gamma
    local gamma_im_vec, omega0_inv, fr, one_minus_fr, omega, raman, shock
    if is_freq_gamma
        gamma_im_vec = im .* medium.gamma
        omega0 = 2π * 3e8 / medium.lambda0
        omega0_inv = 1.0 / omega0
        fr = params.fr
        one_minus_fr = 1.0 - fr
        omega = grid.omega
        raman = params.raman
        shock = params.shock
    end
    
    # Propagation loop
    for i in 2:n_saves
        if progress && i % max(1, n_saves ÷ 10) == 0
            println("Progress: $(round(100*i/n_saves, digits=1))%")
        end
        
        # Half-step linear
        apply_dispersion!(Aw, linop, dz/2)
        
        # For nonlinear step, need to work with physical field
        if is_freq_gamma
            # Remove scaling to get physical field
            Aw_phys = scaling !== nothing ? Aw ./ scaling : Aw
            At_phys = fft_plan * Aw_phys
            
            # Calculate nonlinear operator in frequency domain
            nonlin_w = nonlinear_operator_frequency_dependent(At_phys, gamma_im_vec, omega0_inv, 
                                                              fr, one_minus_fr, omega, fft_plan, ifft_plan,
                                                              RW, raman, shock)
            
            # Apply nonlinear step: exp(i*γ(ω)*...*dz)
            # For M-GNLSE: apply to pseudo-envelope
            if scaling !== nothing
                nonlin_w .*= scaling
            end
            
            # Simple exponential step (approximation)
            @. Aw *= exp(nonlin_w * dz)
        else
            # Transform to time domain
            At = fft_plan * Aw
            
            # Full-step nonlinear (scalar gamma)
            nonlin = nonlinear_operator(At, params, grid, RW)
            @. At *= exp(nonlin * dz)
            
            # Transform to frequency domain
            Aw = ifft_plan * At
        end
        
        # Half-step linear
        apply_dispersion!(Aw, linop, dz/2)
        
        # Save (unscaled physical field)
        if is_freq_gamma && scaling !== nothing
            Aw_phys_save = Aw ./ scaling
            At_out[:, i] = fft_plan * Aw_phys_save
            Aw_out[:, i] = Aw_phys_save
        else
            At_out[:, i] = fft_plan * Aw
            Aw_out[:, i] = Aw
        end
    end
    
    (collect(z_array), At_out, Aw_out)
end
