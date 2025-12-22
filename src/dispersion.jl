"""
    dispersion_operator(grid::Grid, medium::Medium)

Construct the linear dispersion operator L̂(ω) = iβ(ω) - α(ω)/2 for GNLSE propagation.

Computes the frequency-domain operator for the linear part of the GNLSE using Taylor expansion
β(ω) = Σₙ₌₂^∞ (βₙ/n!)(ω-ω₀)ⁿ about the carrier frequency ω₀. The β₀ and β₁ terms are omitted
as they represent global phase and group delay. Array indexing: `betas[1]` = β₂, `betas[2]` = β₃.
Loss α must be in Nepers/m. Returns `Vector{ComplexF64}` of length `grid.N` in units [m⁻¹].

Positive β₂ indicates normal dispersion; negative β₂ indicates anomalous dispersion enabling
soliton propagation. Higher-order terms (β₃, β₄) introduce spectral asymmetry and pulse distortion.
"""
function dispersion_operator(grid::Grid, medium::Medium)
    # grid.omega is the frequency detuning Δω = ω - ω₀
    Δω = grid.omega

    # Initialize dispersion
    beta_omega = zeros(Float64, grid.N)

    # Taylor expansion of dispersion
    # Note: betas are in SI units (s^n/m), omega is in rad/s
    # betas[1] = beta2, betas[2] = beta3, etc. (beta0 and beta1 are omitted)
    for (idx, beta_n) in enumerate(medium.betas)
        n = idx + 1  # betas[1] is beta2, so n = 2, 3, 4, ...
        # Higher order terms: βₙ/n! * Δωⁿ
        # Standard convention: β(ω) = Σ (βₙ/n!) Δω^n where βₙ = d^nβ/dω^n
        beta_omega .+= beta_n .* (Δω .^ n) ./ factorial(n)
    end

    # Add loss term (negative imaginary part)
    # alpha is in natural units: Nepers/m (use convert_loss() to convert from dB)
    # Standard GNLSE: ∂Ã/∂z = +iβ(ω)Ã - (α/2)Ã + nonlinear terms
    linop = im .* beta_omega .- medium.alpha ./ 2

    ComplexF64.(linop)
end

"""
    apply_dispersion!(Aw::Vector{<:Complex}, linop::Vector{<:Complex}, dz::Real)

Apply linear dispersion operator to frequency-domain field in-place via Ã_out(ω) = Ã_in(ω) × exp(L̂(ω) × dz).

Modifies `Aw` directly using zero-allocation broadcasting. Typically called hundreds to thousands of
times per simulation in split-step propagation schemes. Returns `nothing`. O(N) complexity where N is grid size.

  - `Aw`: Frequency-domain field [√W·s], modified in-place
  - `linop`: Linear operator L̂(ω) = iβ(ω) - α(ω)/2 [m⁻¹]
  - `dz`: Propagation step [m]
"""
function apply_dispersion!(Aw::Vector{<:Complex}, linop::Vector{<:Complex}, dz::Real)
    @. Aw *= exp(linop * dz)
    nothing
end

"""
    apply_dispersion(Aw::Vector{<:Complex}, linop::Vector{<:Complex}, dz::Real)

Apply dispersion operator to frequency-domain field (allocating version).

# Arguments

  - `Aw::Vector{<:Complex}`: Frequency-domain field Ã(ω) [√W·s]
  - `linop::Vector{<:Complex}`: Linear dispersion operator L̂(ω) [m⁻¹]
  - `dz::Real`: Propagation step [m]

# Returns

  - `Vector{ComplexF64}`: Propagated frequency-domain field Ã_out(ω)

# Performance Note

This version allocates a new array. For performance-critical loops, use [`apply_dispersion!`](@ref) instead.

# Example

```julia
linop = dispersion_operator(grid, medium)
Aw = ifft(pulse.At)
Aw_propagated = apply_dispersion(Aw, linop, 0.001)
```
"""
function apply_dispersion(Aw::Vector{<:Complex}, linop::Vector{<:Complex}, dz::Real)
    @. Aw * exp(linop * dz)
end
