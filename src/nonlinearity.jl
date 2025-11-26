# ============================================================================
# Optimized nonlinear operators - no conditionals in hot path
# ============================================================================

"""
    nonlinear_operator_kerr(At, gamma_im)

Pure Kerr nonlinearity: iγ|A|²
Optimized - no conditionals.
"""
@inline function nonlinear_operator_kerr(At::Vector{<:Complex}, gamma_im::Complex)
    It = abs2.(At)
    return gamma_im .* It
end

"""
    nonlinear_operator_kerr_raman(At, gamma_im, fr, one_minus_fr, RW, fft_plan, ifft_plan)

Kerr + Raman nonlinearity: iγ[(1-fr)|A|² + fr*(|A|² ⊗ h_R)]
Optimized - no conditionals.
"""
@inline function nonlinear_operator_kerr_raman(At::Vector{<:Complex}, gamma_im::Complex,
                                               fr::Float64, one_minus_fr::Float64,
                                               RW::Vector{<:Complex},
                                               fft_plan, ifft_plan)
    It = abs2.(At)
    It_w = ifft_plan * It
    @. It_w *= RW
    R_term = fft_plan * It_w
    @. R_term = one_minus_fr * It + fr * R_term
    return gamma_im .* R_term
end

"""
    nonlinear_operator_kerr_shock(At, gamma_im, gamma_over_omega0_im, omega, fft_plan, ifft_plan)

Kerr + self-steepening: iγ|A|² + iγ/ω₀ * ∂|A|²/∂t
Optimized - no conditionals.
"""
@inline function nonlinear_operator_kerr_shock(At::Vector{<:Complex}, gamma_im::Complex,
                                               gamma_over_omega0_im::Complex,
                                               omega::Vector{Float64},
                                               fft_plan, ifft_plan)
    It = abs2.(At)
    nonlin = gamma_im .* It
    
    # Shock term
    It_w = ifft_plan * It
    @. It_w *= (im * omega)
    dI_dt = fft_plan * It_w
    @. nonlin += gamma_over_omega0_im * dI_dt
    
    return nonlin
end

"""
    nonlinear_operator_kerr_raman_shock(At, gamma_im, gamma_over_omega0_im, fr, one_minus_fr, 
                                        RW, omega, fft_plan, ifft_plan)

Full nonlinearity: Kerr + Raman + self-steepening
Optimized - no conditionals.
"""
@inline function nonlinear_operator_kerr_raman_shock(At::Vector{<:Complex}, gamma_im::Complex,
                                                     gamma_over_omega0_im::Complex,
                                                     fr::Float64, one_minus_fr::Float64,
                                                     RW::Vector{<:Complex}, omega::Vector{Float64},
                                                     fft_plan, ifft_plan)
    It = abs2.(At)
    
    # Raman
    It_w = ifft_plan * It
    @. It_w *= RW
    R_term = fft_plan * It_w
    @. R_term = one_minus_fr * It + fr * R_term
    
    nonlin = gamma_im .* R_term
    
    # Shock term  
    R_w = ifft_plan * R_term
    @. R_w *= (im * omega)
    dR_dt = fft_plan * R_w
    @. nonlin += gamma_over_omega0_im * dR_dt
    
    return nonlin
end

"""
    nonlinear_operator(At::Vector{<:Complex}, params::SimParams, grid::Grid, RW::Union{Vector{<:Complex},Nothing}=nothing)

Calculate the nonlinear operator for GNLSE with scalar gamma.
Dispatcher that calls specialized optimized functions based on enabled effects.

# Arguments
- `At::Vector{<:Complex}`: Time-domain field
- `params::SimParams`: Simulation parameters
- `grid::Grid`: Time/frequency grid
- `RW::Union{Vector{<:Complex},Nothing}`: (Optional) Frequency-domain Raman response

# Returns
- `Vector{ComplexF64}`: Nonlinear term ∂A/∂z|_nonlinear
"""
function nonlinear_operator(At::Vector{<:Complex}, params::SimParams, grid::Grid,
                           RW::Union{Vector{<:Complex},Nothing}=nothing)
    # Ensure scalar gamma
    gamma = params.medium.gamma
    if gamma isa Vector
        error("Use nonlinear_operator_frequency_dependent for vector gamma")
    end
    
    # Pre-compute constants
    gamma_im = im * gamma
    omega0 = 2π * 3e8 / params.medium.lambda0
    gamma_over_omega0_im = im * gamma / omega0
    
    # Create FFT plans once (cached)
    fft_plan = plan_fft(At)
    ifft_plan = plan_ifft(At)
    
    # Dispatch to specialized function (compiler eliminates dead branches)
    if params.raman && RW !== nothing
        if params.shock
            return nonlinear_operator_kerr_raman_shock(At, gamma_im, gamma_over_omega0_im,
                                                       params.fr, 1.0 - params.fr,
                                                       RW, grid.omega, fft_plan, ifft_plan)
        else
            return nonlinear_operator_kerr_raman(At, gamma_im, params.fr, 1.0 - params.fr,
                                                 RW, fft_plan, ifft_plan)
        end
    else
        if params.shock
            return nonlinear_operator_kerr_shock(At, gamma_im, gamma_over_omega0_im,
                                                grid.omega, fft_plan, ifft_plan)
        else
            return nonlinear_operator_kerr(At, gamma_im)
        end
    end
end

# ============================================================================
# Optimized frequency-dependent nonlinear operators
# ============================================================================

"""
    nonlinear_freq_gamma_kerr(At, gamma_im_vec, fft_plan, ifft_plan)

Pure Kerr with frequency-dependent γ(ω)
"""
@inline function nonlinear_freq_gamma_kerr(At::Vector{<:Complex}, 
                                           gamma_im_vec::Vector{<:Complex},
                                           fft_plan, ifft_plan)
    It = abs2.(At)
    nonlin_t = At .* It
    nonlin_w = ifft_plan * nonlin_t
    @. nonlin_w *= gamma_im_vec
    return nonlin_w
end

"""
    nonlinear_freq_gamma_kerr_raman(At, gamma_im_vec, fr, one_minus_fr, RW, fft_plan, ifft_plan)

Kerr + Raman with frequency-dependent γ(ω)
"""
@inline function nonlinear_freq_gamma_kerr_raman(At::Vector{<:Complex},
                                                 gamma_im_vec::Vector{<:Complex},
                                                 fr::Float64, one_minus_fr::Float64,
                                                 RW::Vector{<:Complex},
                                                 fft_plan, ifft_plan)
    It = abs2.(At)
    It_w = ifft_plan * It
    @. It_w *= RW
    R_term = fft_plan * It_w
    @. R_term = one_minus_fr * It + fr * R_term
    
    nonlin_t = At .* R_term
    nonlin_w = ifft_plan * nonlin_t
    @. nonlin_w *= gamma_im_vec
    return nonlin_w
end

"""
    nonlinear_freq_gamma_kerr_shock(At, gamma_im_vec, omega0_inv, omega, fft_plan, ifft_plan)

Kerr + shock with frequency-dependent γ(ω)
"""
@inline function nonlinear_freq_gamma_kerr_shock(At::Vector{<:Complex},
                                                 gamma_im_vec::Vector{<:Complex},
                                                 omega0_inv::Float64,
                                                 omega::Vector{Float64},
                                                 fft_plan, ifft_plan)
    It = abs2.(At)
    nonlin_t = At .* It
    
    # Shock term
    temp_w = ifft_plan * nonlin_t
    temp_w_shock = copy(temp_w)
    @. temp_w_shock *= omega
    d_nonlin_dt = fft_plan * temp_w_shock
    @. nonlin_t += d_nonlin_dt * omega0_inv
    
    # Apply γ(ω) in frequency domain
    nonlin_w = ifft_plan * nonlin_t  # Reuse temp_w
    @. nonlin_w *= gamma_im_vec
    return nonlin_w
end

"""
    nonlinear_freq_gamma_kerr_raman_shock(At, gamma_im_vec, omega0_inv, fr, one_minus_fr,
                                          RW, omega, fft_plan, ifft_plan)

Full nonlinearity with frequency-dependent γ(ω)
"""
@inline function nonlinear_freq_gamma_kerr_raman_shock(At::Vector{<:Complex},
                                                       gamma_im_vec::Vector{<:Complex},
                                                       omega0_inv::Float64,
                                                       fr::Float64, one_minus_fr::Float64,
                                                       RW::Vector{<:Complex},
                                                       omega::Vector{Float64},
                                                       fft_plan, ifft_plan)
    It = abs2.(At)
    It_w = ifft_plan * It
    @. It_w *= RW
    R_term = fft_plan * It_w
    @. R_term = one_minus_fr * It + fr * R_term
    
    nonlin_t = At .* R_term
    
    # Shock term
    temp_w = ifft_plan * nonlin_t
    temp_w_shock = copy(temp_w)
    @. temp_w_shock *= omega
    d_nonlin_dt = fft_plan * temp_w_shock
    @. nonlin_t += d_nonlin_dt * omega0_inv
    
    # Apply γ(ω) in frequency domain
    nonlin_w = ifft_plan * nonlin_t
    @. nonlin_w *= gamma_im_vec
    return nonlin_w
end

"""
    nonlinear_operator_frequency_dependent(At::Vector{<:Complex}, Aw::Vector{<:Complex}, 
                                          params::SimParams, grid::Grid, 
                                          RW::Union{Vector{<:Complex},Nothing}=nothing)

Calculate the nonlinear operator for M-GNLSE with frequency-dependent gamma.
Dispatcher that calls specialized optimized functions.

Implements the Lægsgaard (2007) pseudo-envelope method:
1. Field is pre-scaled by [Aeff(ω₀)/Aeff(ω)]^(1/4) (applied externally)
2. Nonlinearity computed in time domain: N(t) = |A(t)|² * A(t)
3. Result transformed to frequency domain and multiplied by γ(ω)

This avoids the convolution problem by working with the pseudo-envelope C(z,ω).

# Arguments
- `At::Vector{<:Complex}`: Time-domain field (already scaled if scaling provided)
- `Aw::Vector{<:Complex}`: Frequency-domain field (for applying γ(ω))
- `params::SimParams`: Simulation parameters (with vector gamma)
- `grid::Grid`: Time/frequency grid
- `RW::Union{Vector{<:Complex},Nothing}`: (Optional) Frequency-domain Raman response

# Returns
- `Vector{ComplexF64}`: Nonlinear term in frequency domain

# References
J. Lægsgaard, "Mode profile dispersion in the generalized nonlinear Schrödinger equation,"
Opt. Express 15, 16110-16123 (2007).
"""
function nonlinear_operator_frequency_dependent(At::Vector{<:Complex}, gamma_im_vec::Vector{<:Complex},
                                               omega0_inv::Float64, fr::Float64, one_minus_fr::Float64,
                                               omega::Vector{Float64}, fft_plan, ifft_plan,
                                               RW::Union{Vector{<:Complex},Nothing},
                                               raman::Bool, shock::Bool)
    # Dispatch to specialized function (all parameters pre-computed)
    if raman && RW !== nothing
        if shock
            return nonlinear_freq_gamma_kerr_raman_shock(At, gamma_im_vec, omega0_inv,
                                                         fr, one_minus_fr,
                                                         RW, omega, fft_plan, ifft_plan)
        else
            return nonlinear_freq_gamma_kerr_raman(At, gamma_im_vec, fr, one_minus_fr,
                                                   RW, fft_plan, ifft_plan)
        end
    else
        if shock
            return nonlinear_freq_gamma_kerr_shock(At, gamma_im_vec, omega0_inv,
                                                   omega, fft_plan, ifft_plan)
        else
            return nonlinear_freq_gamma_kerr(At, gamma_im_vec, fft_plan, ifft_plan)
        end
    end
end

"""
    apply_nonlinearity!(At::Vector{<:Complex}, nonlin::Vector{<:Complex}, dz::Real)

Apply nonlinear operator in-place (simple exponential step).

# Arguments
- `At::Vector{<:Complex}`: Time-domain field (modified in-place)
- `nonlin::Vector{<:Complex}`: Nonlinear operator
- `dz::Real`: Propagation step [m]
"""
function apply_nonlinearity!(At::Vector{<:Complex}, nonlin::Vector{<:Complex}, dz::Real)
    @. At *= exp(nonlin * dz)
    nothing
end
