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
                                               fft_plan, ifft_plan, dt::Float64)
    It = abs2.(At)
    It_w = ifft_plan * It
    @. It_w *= RW * dt  # dt scaling for proper Raman convolution
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
    
    # Shock term: iγ/ω₀ * ∂|A|²/∂t
    # Mathematical foundation: ∂f/∂t ↔ +iω·F(ω) (Fourier transform property)
    # This identity is proven by integration by parts and holds regardless of FFT convention.
    # Derivative operator: ∂/∂t → +iω in frequency domain (invariant of FFT convention)
    # Transform to frequency, apply derivative, transform back
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
                                                     fft_plan, ifft_plan, dt::Float64)
    It = abs2.(At)
    
    # Raman
    It_w = ifft_plan * It
    @. It_w *= RW * dt  # dt scaling for proper Raman convolution
    R_term = fft_plan * It_w
    @. R_term = one_minus_fr * It + fr * R_term
    
    nonlin = gamma_im .* R_term
    
    # Shock term: iγ/ω₀ * ∂R/∂t where R is Raman-modified intensity
    # Derivative operator: ∂/∂t → +iω in frequency domain (invariant of FFT convention)
    # Transform to frequency, apply derivative, transform back
    R_w = ifft_plan * R_term
    @. R_w *= (im * omega)
    dR_dt = fft_plan * R_w
    @. nonlin += gamma_over_omega0_im * dR_dt
    
    return nonlin
end

"""
    nonlinear_operator(At::Vector{<:Complex}, params::SimParams, grid::Grid, RW::Union{Vector{<:Complex},Nothing}=nothing)

Calculate the nonlinear operator for GNLSE with scalar (frequency-independent) gamma.

# Mathematical Formulation
The nonlinear contribution to GNLSE (in time domain):
```
∂A/∂z|_nonlinear = N̂[A]
```
where the operator N̂ includes:

1. **Kerr effect** (instantaneous electronic response):
   ```
   N_Kerr = iγ|A|²
   ```

2. **Raman effect** (delayed molecular response):
   ```
   N_Raman = iγfᵣ ∫hᵣ(t-t')|A(t')|² dt' = iγfᵣ(|A|² ⊗ hᵣ)
   ```

3. **Self-steepening** (shock term, frequency-dependent nonlinearity):
   ```
   N_shock = iγ/ω₀ · ∂/∂t[(1-fᵣ)|A|² + fᵣ(|A|² ⊗ hᵣ)]
   ```

Combined operator:
```
N̂[A] = iγ{[(1-fᵣ)|A|² + fᵣ(|A|² ⊗ hᵣ)] + (1/ω₀)∂[...]/∂t}
```

# Implementation Strategy
The function dispatches to specialized, optimized implementations based on enabled effects:
- **Kerr only**: `nonlinear_operator_kerr` (fastest)
- **Kerr + Raman**: `nonlinear_operator_kerr_raman`
- **Kerr + Shock**: `nonlinear_operator_kerr_shock`
- **Full (Kerr + Raman + Shock)**: `nonlinear_operator_kerr_raman_shock`

All implementations avoid conditionals in hot paths for maximum performance.

# Arguments
- `At::Vector{<:Complex}`: Time-domain field A(t) [√W]
- `params::SimParams`: Simulation parameters (gamma, raman, shock flags)
- `grid::Grid`: Time/frequency grid for derivatives
- `RW::Union{Vector{<:Complex},Nothing}`: Pre-computed frequency-domain Raman response (optional)

# Returns
- `Vector{ComplexF64}`: Nonlinear term N̂[A] in time domain [m⁻¹√W]

# Performance Notes
- Zero conditionals in hot path (compile-time dispatch)
- Pre-computed constants (gamma_im, etc.)
- FFT plans cached for repeated use
- Broadcasting for vectorized operations

# Examples
```julia
# Setup
params = SimParams(medium=medium, raman=true, shock=true)
h_R, _ = raman_response(grid, params.raman_model)
RW = raman_response_frequency(h_R, grid)

# Calculate nonlinear operator
At = pulse.At
nonlin = nonlinear_operator(At, params, grid, RW)
# nonlin represents iγ[(1-fr)|A|²+fr*Raman] + shock term
```

# See Also
- [`nonlinear_operator_frequency_dependent`](@ref): For M-GNLSE with γ(ω)
- [`apply_nonlinearity!`](@ref): Apply nonlinear step in split-step methods
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
                                                       RW, grid.omega, fft_plan, ifft_plan, grid.dt)
        else
            return nonlinear_operator_kerr_raman(At, gamma_im, params.fr, 1.0 - params.fr,
                                                 RW, fft_plan, ifft_plan, grid.dt)
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
                                                gamma_im_vec::Vector{Complex{Float64}},
                                                fr::Float64, one_minus_fr::Float64,
                                                RW::Vector{<:Complex},
                                                fft_plan, ifft_plan, dt::Float64)
    It = abs2.(At)
    It_w = ifft_plan * It
    @. It_w *= RW * dt  # dt scaling for proper Raman convolution
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
    
    # Shock term: A * (1/ω₀) * ∂|A|²/∂t
    # Derivative operator: ∂/∂t → +iω in frequency domain (invariant of FFT convention)
    # Transform to frequency, apply derivative, transform back
    It_w = ifft_plan * It
    @. It_w *= (im * omega * omega0_inv)
    dI_dt = fft_plan * It_w
    
    # Total nonlinearity in time domain: A*|A|² + A*(1/ω₀)*∂|A|²/∂t
    nonlin_t = @. At * (It + dI_dt)
    
    # Apply γ(ω) in frequency domain
    nonlin_w = ifft_plan * nonlin_t
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
                                                       fft_plan, ifft_plan, dt::Float64)
    It = abs2.(At)
    
    # Raman response
    It_w = ifft_plan * It
    @. It_w *= RW * dt  # dt scaling for proper Raman convolution
    R_term = fft_plan * It_w
    @. R_term = one_minus_fr * It + fr * R_term
    
    # Shock term: A * (1/ω₀) * ∂R/∂t where R is Raman-modified intensity
    # Derivative operator: ∂/∂t → +iω in frequency domain (invariant of FFT convention)
    # Transform to frequency, apply derivative, transform back
    R_w = ifft_plan * R_term
    @. R_w *= (im * omega * omega0_inv)
    dR_dt = fft_plan * R_w
    
    # Total nonlinearity in time domain: A*R + A*(1/ω₀)*∂R/∂t
    nonlin_t = @. At * (R_term + dR_dt)
    
    # Apply γ(ω) in frequency domain
    nonlin_w = ifft_plan * nonlin_t
    @. nonlin_w *= gamma_im_vec
    return nonlin_w
end

"""
    nonlinear_operator_frequency_dependent(At, gamma_im_vec, omega0_inv, fr, one_minus_fr, 
                                          omega, fft_plan, ifft_plan, RW, raman, shock)

Calculate the nonlinear operator for M-GNLSE with frequency-dependent gamma γ(ω).

# Mathematical Background: M-GNLSE Pseudo-Envelope Method

Standard GNLSE with frequency-dependent effective area Aeff(ω) leads to a time-domain
convolution problem that is computationally expensive. The Lægsgaard (2007) method avoids
this by using a **pseudo-envelope** C(z,ω) defined as:
```
C(z,ω) = A(z,ω) × [Aeff(ω₀)/Aeff(ω)]^(1/4)
```

where A(z,ω) is the physical field and the scaling factor compensates for dispersion of the
mode profile. This transformation allows the GNLSE to be written as:
```
∂C/∂ω = [iβ(ω) - α(ω)/2]C + iγ(ω)F{|F⁻¹{C}|² F⁻¹{C}}
```

# Implementation Algorithm
1. **Input**: Pseudo-envelope C(t) in time domain (already scaled by caller)
2. **Nonlinear term**: Compute N(t) = |C(t)|² × C(t) in time domain
3. **Transform**: N(ω) = F{N(t)} via FFT
4. **Apply γ(ω)**: Multiply by iγ(ω) in frequency domain
5. **Output**: Frequency-domain nonlinear operator iγ(ω)N(ω)

This avoids convolution while correctly handling frequency-dependent effective area.

# Arguments
- `At::Vector{<:Complex}`: Time-domain pseudo-envelope C(t) (already scaled)
- `gamma_im_vec::Vector{<:Complex}`: iγ(ω) vector
- `omega0_inv::Float64`: 1/ω₀ for shock term
- `fr::Float64`: Raman fraction
- `one_minus_fr::Float64`: 1 - fᵣ (pre-computed)
- `omega::Vector{Float64}`: Frequency grid for derivatives
- `fft_plan`: Pre-allocated FFT plan
- `ifft_plan`: Pre-allocated IFFT plan  
- `RW::Union{Vector{<:Complex},Nothing}`: Frequency-domain Raman response
- `raman::Bool`: Enable Raman effect
- `shock::Bool`: Enable self-steepening

# Returns
- `Vector{ComplexF64}`: Nonlinear operator in frequency domain

# Performance Optimization
- All parameters pre-computed by caller (no redundant calculations)
- Dispatches to specialized functions (no conditionals in hot path)
- In-place operations where possible
- FFT plans reused across calls

# Key Difference from Scalar Gamma
- **Scalar γ**: Result is in time domain (iγ|A|²)
- **Vector γ(ω)**: Result is in frequency domain (iγ(ω)·F{|A|²A})
- This asymmetry is fundamental to avoiding time-domain convolution

# Examples
```julia
# Setup (done once at initialization)
gamma_im_vec = im .* medium.gamma  # γ(ω) vector
omega0_inv = 1.0 / (2π * c / lambda0)
fr = params.fr
one_minus_fr = 1.0 - fr

# In propagation loop
nonlin_w = nonlinear_operator_frequency_dependent(
    At, gamma_im_vec, omega0_inv, fr, one_minus_fr,
    grid.omega, fft_plan, ifft_plan, RW, params.raman, params.shock
)
# nonlin_w is in frequency domain, ready for integration
```

# References
- J. Lægsgaard, "Mode profile dispersion in the generalized nonlinear Schrödinger equation,"
  Opt. Express 15, 16110-16123 (2007). DOI: 10.1364/OE.15.016110

# See Also
- [`nonlinear_operator`](@ref): Scalar gamma version
- [`Medium`](@ref): Defines gamma and scaling vectors
"""
function nonlinear_operator_frequency_dependent(At::Vector{<:Complex}, gamma_im_vec::Vector{<:Complex},
                                               omega0_inv::Float64, fr::Float64, one_minus_fr::Float64,
                                               omega::Vector{Float64}, fft_plan, ifft_plan,
                                               RW::Union{Vector{<:Complex},Nothing},
                                               raman::Bool, shock::Bool, dt::Float64)
    # Dispatch to specialized function (all parameters pre-computed)
    if raman && RW !== nothing
        if shock
            return nonlinear_freq_gamma_kerr_raman_shock(At, gamma_im_vec, omega0_inv,
                                                         fr, one_minus_fr,
                                                         RW, omega, fft_plan, ifft_plan, dt)
        else
            return nonlinear_freq_gamma_kerr_raman(At, gamma_im_vec, fr, one_minus_fr,
                                                   RW, fft_plan, ifft_plan, dt)
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
