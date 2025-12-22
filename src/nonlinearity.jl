# ============================================================================
# Nonlinear operators - FiberNlse approach with model struct
# ============================================================================

using FFTW

"""
    PhysicsModel

Precomputed physics operators and FFT plans for GNLSE propagation.

# Fields
- `dispersion_term`: D̂(ω) = -α/2 + i∑(βₙ/n!)(iω)ⁿ
- `fftp`, `ifftp`: Forward and inverse FFT plans
- `γ`: Nonlinear coefficient [1/(W·m)] (scalar or frequency-dependent vector)
- `ω`: Angular frequency grid [rad/s]
- `ω0`: Central angular frequency [rad/s]
- `dt`: Time step [s]
- `fr`: Raman fraction
- `raman_freq_response`: h̃ᵣ(ω) in frequency domain
- `nonlinear_function`: Selected nonlinearity operator
"""
struct PhysicsModel
    dispersion_term::Vector{ComplexF64}
    fftp::Any  # FFTW plan type
    ifftp::Any  # FFTW plan type
    γ::Union{Float64, Vector{ComplexF64}}
    ω::Vector{Float64}
    ω0::Float64
    dt::Float64
    fr::Float64
    raman_freq_response::Union{Vector{ComplexF64}, Nothing}
    nonlinear_function::Function
end

# ============================================================================
# Nonlinear operators following FiberNlse pattern
# Each takes (At, model) and returns frequency-domain nonlinearity
# ============================================================================

"""
    kerr_only(At, model)

Pure Kerr nonlinearity (SPM only).
N̂[A] = iγ·FFT{A·|A|²}
"""
@inline function kerr_only(At::Vector{<:Complex}, model::PhysicsModel)
    return (1.0im * model.γ) .* (model.fftp * (At .* abs2.(At)))
end

"""
    kerr_shock(At, model)

Kerr + self-steepening (shock term).
N̂[A] = iγ(1 - ω/ω₀)·FFT{A·|A|²}
"""
@inline function kerr_shock(At::Vector{<:Complex}, model::PhysicsModel)
    return (1.0im * model.γ) .* (1 .- model.ω ./ model.ω0) .*
           (model.fftp * (At .* abs2.(At)))
end

"""
    kerr_raman(At, model)

Kerr + Raman (no self-steepening).
N̂[A] = iγ·FFT{A·[(1-fᵣ)|A|² + fᵣ(|A|² ⊗ hᵣ)]}
"""
@inline function kerr_raman(At::Vector{<:Complex}, model::PhysicsModel)
    IT = abs2.(At)
    # Raman convolution: fᵣ dt ∫ hᵣ(t')·|A(t-t')|² dt'
    RS =
        model.dt *
        model.fr *
        (model.ifftp * ((model.fftp * IT) .* model.raman_freq_response))
    # Combined: (1-fᵣ)|A|² + fᵣ·RS
    return (1.0im * model.γ) .* (model.fftp * (At .* ((1.0 - model.fr) .* IT .+ RS)))
end

"""
    kerr_raman_shock(At, model)

Full nonlinearity: Kerr + Raman + self-steepening.
N̂[A] = iγ(1 - ω/ω₀)·FFT{A·[(1-fᵣ)|A|² + fᵣ(|A|² ⊗ hᵣ)]}
"""
@inline function kerr_raman_shock(At::Vector{<:Complex}, model::PhysicsModel)
    IT = abs2.(At)
    RS =
        model.dt *
        model.fr *
        (model.ifftp * ((model.fftp * IT) .* model.raman_freq_response))
    return (1.0im * model.γ) .* (1 .- model.ω ./ model.ω0) .*
           (model.fftp * (At .* ((1.0 - model.fr) .* IT .+ RS)))
end

# ============================================================================
# Frequency-dependent nonlinear operators (M-GNLSE)
# Following Lægsgaard (2007) pseudo-envelope method
# ============================================================================

"""
    kerr_only_freq(At, model)

Pure Kerr with frequency-dependent γ(ω).
N̂[A] = F⁻¹[γ̃(ω)·F{A·|A|²}] where γ̃(ω) = iγ(ω)
"""
@inline function kerr_only_freq(At::Vector{<:Complex}, model::PhysicsModel)
    # Transform nonlinear product to frequency domain
    Aw_nonlin = model.fftp * (At .* abs2.(At))
    # Apply frequency-dependent gamma (already includes i factor)
    @. Aw_nonlin *= model.γ
    return Aw_nonlin
end

"""
    kerr_shock_freq(At, model)

Kerr + self-steepening with frequency-dependent γ(ω).
N̂[A] = F⁻¹[γ̃(ω)(1 - ω/ω₀)·F{A·|A|²}]
"""
@inline function kerr_shock_freq(At::Vector{<:Complex}, model::PhysicsModel)
    Aw_nonlin = model.fftp * (At .* abs2.(At))
    @. Aw_nonlin *= model.γ * (1 - model.ω / model.ω0)
    return Aw_nonlin
end

"""
    kerr_raman_freq(At, model)

Kerr + Raman with frequency-dependent γ(ω).
N̂[A] = F⁻¹[γ̃(ω)·F{A·[(1-fᵣ)|A|² + fᵣ(|A|² ⊗ hᵣ)]}]
"""
@inline function kerr_raman_freq(At::Vector{<:Complex}, model::PhysicsModel)
    IT = abs2.(At)
    RS =
        model.dt *
        model.fr *
        (model.ifftp * ((model.fftp * IT) .* model.raman_freq_response))
    Aw_nonlin = model.fftp * (At .* ((1.0 - model.fr) .* IT .+ RS))
    @. Aw_nonlin *= model.γ
    return Aw_nonlin
end

"""
    kerr_raman_shock_freq(At, model)

Full nonlinearity with frequency-dependent γ(ω): Kerr + Raman + shock.
N̂[A] = F⁻¹[γ̃(ω)(1 - ω/ω₀)·F{A·[(1-fᵣ)|A|² + fᵣ(|A|² ⊗ hᵣ)]}]
"""
@inline function kerr_raman_shock_freq(At::Vector{<:Complex}, model::PhysicsModel)
    IT = abs2.(At)
    RS =
        model.dt *
        model.fr *
        (model.ifftp * ((model.fftp * IT) .* model.raman_freq_response))
    Aw_nonlin = model.fftp * (At .* ((1.0 - model.fr) .* IT .+ RS))
    @. Aw_nonlin *= model.γ * (1 - model.ω / model.ω0)
    return Aw_nonlin
end

"""
    select_nonlinearity(shock::Bool, raman::Bool, freq_dependent::Bool)

Select nonlinear operator at compile time for zero-overhead dispatch.

# Arguments
- `shock`: Include self-steepening
- `raman`: Include Raman scattering
- `freq_dependent`: Use frequency-dependent γ(ω) (M-GNLSE)

Returns function reference to appropriate nonlinear operator.
"""
function select_nonlinearity(shock::Bool, raman::Bool, freq_dependent::Bool)
    if freq_dependent
        # M-GNLSE operators with γ(ω)
        return if shock && raman
            kerr_raman_shock_freq
        elseif shock
            kerr_shock_freq
        elseif raman
            kerr_raman_freq
        else
            kerr_only_freq
        end
    else
        # Standard GNLSE operators with scalar γ
        return if shock && raman
            kerr_raman_shock
        elseif shock
            kerr_shock
        elseif raman
            kerr_raman
        else
            kerr_only
        end
    end
end

"""
    build_physics_model(grid::Grid, params::SimParams)

Construct `PhysicsModel` with precomputed operators and FFT plans.
Automatically detects scalar γ (GNLSE) or frequency-dependent γ(ω) (M-GNLSE).
FFT plans use FFTW.MEASURE for optimized execution.

Returns `PhysicsModel` struct with all precomputed physics operators.
"""
function build_physics_model(grid::Grid, params::SimParams)
    # Create FFT plans with MEASURE flag for optimized transforms
    # Takes longer to plan but 10-50% faster execution (critical for repeated use)
    dummy = zeros(ComplexF64, grid.N)
    fftp = plan_fft(dummy; flags=FFTW.MEASURE)
    ifftp = plan_ifft(dummy; flags=FFTW.MEASURE)

    # Dispersion operator: D̂(ω) = -α/2 + i∑(βₙ/n!)(iω)ⁿ
    D = dispersion_operator(grid, params.medium)

    # Raman response if needed
    raman_freq_resp = nothing
    if params.raman
        h_R, _ = raman_response(grid, params.raman_model)
        raman_freq_resp = raman_response_frequency(h_R, grid)
    end

    # Central frequency from grid (already computed)
    ω0 = grid.omega0

    # Handle scalar or frequency-dependent gamma
    γ_raw = params.medium.gamma
    freq_dependent = γ_raw isa AbstractVector

    γ = if freq_dependent
        # M-GNLSE: γ(ω) stored as complex vector with i factor pre-applied
        # Check grid size matches
        length(γ_raw) == grid.N || error(
            "Frequency-dependent gamma must have length $(grid.N), got $(length(γ_raw))",
        )

        # Pre-multiply by i for efficiency: γ̃(ω) = i·γ(ω)
        Complex{Float64}.(1.0im .* γ_raw)
    else
        # Standard GNLSE: scalar gamma
        Float64(γ_raw)
    end

    # Select nonlinear operator (compile-time dispatch, no runtime branching)
    nonlin_fn = select_nonlinearity(params.shock, params.raman, freq_dependent)

    return PhysicsModel(
        D, fftp, ifftp, γ, grid.omega, ω0, grid.dt, params.fr, raman_freq_resp, nonlin_fn
    )
end

"""
    nonlinear_operator(At::Vector{<:Complex}, grid::Grid, params::SimParams)

Compute nonlinear operator N̂[A] = iγ[(1-fᵣ)|A|² + fᵣ(|A|² ⊗ hᵣ) + (1/ω₀)∂[...]/∂t].
Legacy interface wrapping `PhysicsModel` approach for backward compatibility.

# Arguments
- `At`: Time-domain field amplitude
- `grid`: Grid structure with FFT plans, dt, omega
- `params`: SimParams with raman/shock flags and medium parameters

Returns nonlinear operator in frequency domain for split-step integration.
"""
function nonlinear_operator(At::Vector{<:Complex}, grid::Grid, params::SimParams)
    # Build model on the fly for legacy compatibility
    # In new solver, model is built once and reused
    model = build_physics_model(grid, params)
    return model.nonlinear_function(At, model)
end

# ============================================================================
# Optimized frequency-dependent nonlinear operators
# ============================================================================

"""
    nonlinear_freq_gamma_kerr(At, gamma_im_vec, fft_plan, ifft_plan)

Pure Kerr nonlinearity with frequency-dependent γ(ω): N̂[A] = F⁻¹[γ̃(ω)·F{A·|A|²}].
"""
@inline function nonlinear_freq_gamma_kerr(
    At::Vector{<:Complex}, gamma_im_vec::Vector{<:Complex}, fft_plan, ifft_plan
)
    It = abs2.(At)
    nonlin_t = At .* It
    nonlin_w = ifft_plan * nonlin_t
    @. nonlin_w *= gamma_im_vec
    return nonlin_w
end

"""
    nonlinear_freq_gamma_kerr_raman(At, gamma_im_vec, fr, one_minus_fr, RW, fft_plan, ifft_plan, dt)

Kerr + Raman with frequency-dependent γ(ω): N̂[A] = F⁻¹[γ̃(ω)·F{A·[(1-fᵣ)|A|² + fᵣ(|A|² ⊗ hᵣ)]}].
"""
@inline function nonlinear_freq_gamma_kerr_raman(
    At::Vector{<:Complex},
    gamma_im_vec::Vector{Complex{Float64}},
    fr::Float64,
    one_minus_fr::Float64,
    RW::Vector{<:Complex},
    fft_plan,
    ifft_plan,
    dt::Float64,
)
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

Kerr + self-steepening with frequency-dependent γ(ω): N̂[A] = F⁻¹[γ̃(ω)(1-ω/ω₀)·F{A·|A|²}].
"""
@inline function nonlinear_freq_gamma_kerr_shock(
    At::Vector{<:Complex},
    gamma_im_vec::Vector{<:Complex},
    omega0_inv::Float64,
    omega::Vector{Float64},
    fft_plan,
    ifft_plan,
)
    It = abs2.(At)

    # Shock term: A * (1/ω₀) * ∂|A|²/∂t
    # With inverted FFT: ifft(time) → freq, fft(freq) → time
    # Derivative: multiply by -iω (since ifft inverts the sign)
    It_w = ifft_plan * It
    @. It_w *= (-im * omega * omega0_inv)
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
                                          RW, omega, fft_plan, ifft_plan, dt)

Full nonlinearity with frequency-dependent γ(ω): Kerr + Raman + self-steepening.
N̂[A] = F⁻¹[γ̃(ω)(1-ω/ω₀)·F{A·[(1-fᵣ)|A|² + fᵣ(|A|² ⊗ hᵣ)]}].
"""
@inline function nonlinear_freq_gamma_kerr_raman_shock(
    At::Vector{<:Complex},
    gamma_im_vec::Vector{<:Complex},
    omega0_inv::Float64,
    fr::Float64,
    one_minus_fr::Float64,
    RW::Vector{<:Complex},
    omega::Vector{Float64},
    fft_plan,
    ifft_plan,
    dt::Float64,
)
    It = abs2.(At)

    # Raman response
    It_w = ifft_plan * It
    @. It_w *= RW * dt  # dt scaling for proper Raman convolution
    R_term = fft_plan * It_w
    @. R_term = one_minus_fr * It + fr * R_term

    # Shock term: A * (1/ω₀) * ∂R/∂t where R is Raman-modified intensity
    # With inverted FFT: ifft(time) → freq, fft(freq) → time
    # Derivative: multiply by -iω (since ifft inverts the sign)
    R_w = ifft_plan * R_term
    @. R_w *= (-im * omega * omega0_inv)
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
                                          omega, fft_plan, ifft_plan, RW, raman, shock, dt)

Compute M-GNLSE nonlinear operator with frequency-dependent γ(ω).
Uses Lægsgaard (2007) pseudo-envelope method: ∂C/∂z = [iβ(ω) - α/2]C + iγ(ω)F{|F⁻¹{C}|²F⁻¹{C}}.

# Arguments
- `At`: Time-domain pseudo-envelope C(t)
- `gamma_im_vec`: iγ(ω) vector [1/(W·m)]
- `omega0_inv`: 1/ω₀ [s/rad] for shock term
- `fr`: Raman fraction
- `one_minus_fr`: 1 - fᵣ (precomputed)
- `omega`: Frequency grid [rad/s]
- `fft_plan`, `ifft_plan`: FFT plans
- `RW`: h̃ᵣ(ω) Raman response or `nothing`
- `raman`, `shock`: Effect flags
- `dt`: Time step [s]

Returns nonlinear operator in frequency domain.
Reference: Lægsgaard, Opt. Express 15, 16110 (2007).
"""
function nonlinear_operator_frequency_dependent(
    At::Vector{<:Complex},
    gamma_im_vec::Vector{<:Complex},
    omega0_inv::Float64,
    fr::Float64,
    one_minus_fr::Float64,
    omega::Vector{Float64},
    fft_plan,
    ifft_plan,
    RW::Union{Vector{<:Complex}, Nothing},
    raman::Bool,
    shock::Bool,
    dt::Float64,
)
    # Dispatch to specialized function (all parameters pre-computed)
    if raman && RW !== nothing
        if shock
            return nonlinear_freq_gamma_kerr_raman_shock(
                At,
                gamma_im_vec,
                omega0_inv,
                fr,
                one_minus_fr,
                RW,
                omega,
                fft_plan,
                ifft_plan,
                dt,
            )
        else
            return nonlinear_freq_gamma_kerr_raman(
                At, gamma_im_vec, fr, one_minus_fr, RW, fft_plan, ifft_plan, dt
            )
        end
    else
        if shock
            return nonlinear_freq_gamma_kerr_shock(
                At, gamma_im_vec, omega0_inv, omega, fft_plan, ifft_plan
            )
        else
            return nonlinear_freq_gamma_kerr(At, gamma_im_vec, fft_plan, ifft_plan)
        end
    end
end

"""
    apply_nonlinearity!(At::Vector{<:Complex}, nonlin::Vector{<:Complex}, dz::Real)

Apply nonlinear operator in-place via exponential step: Aₜ ← Aₜ·exp(N̂·dz).

# Arguments
- `At`: Time-domain field (modified in-place)
- `nonlin`: Nonlinear operator
- `dz`: Propagation step [m]
"""
function apply_nonlinearity!(At::Vector{<:Complex}, nonlin::Vector{<:Complex}, dz::Real)
    @. At *= exp(nonlin * dz)
    nothing
end
