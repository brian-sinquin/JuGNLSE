"""
    dispersion_operator(grid::Grid, medium::Medium)

Construct the linear dispersion operator in the frequency domain for GNLSE propagation.

# Mathematical Formulation

The dispersion operator represents the linear part of the GNLSE:

```
∂A/∂z|_linear = L̂[A] = [iβ(ω) - α(ω)/2] Ã(ω)
```

where the propagation constant β(ω) is given by the Taylor expansion:

```
β(ω) = β₀ + β₁(ω-ω₀) + Σₙ₌₂^∞ (βₙ/n!) (ω-ω₀)ⁿ
```

# Implementation Details

  - `grid.omega` contains frequency detuning Δω = ω - ω₀ [rad/s]
  - `medium.betas[n]` stores βₙ₊₁ (i.e., `betas[1]` = β₂, `betas[2]` = β₃, etc.)
  - β₀ and β₁ terms are omitted (global phase and group delay)
  - Loss term: α must be in Nepers/m (use `convert_loss()` for dB/km conversion)

# Arguments

  - `grid::Grid`: Time-frequency grid containing Δω array
  - `medium::Medium`: Medium parameters with beta coefficients and loss

# Returns

  - `Vector{ComplexF64}`: Linear operator L̂(ω) = iβ(ω) - α(ω)/2

# Example

```julia
grid = create_grid(2^12, 10e-12, 835e-9)
# Convert loss from dB/km to Nepers/m if needed
alpha_dbkm = 0.2  # dB/km
alpha_npm = convert_loss(alpha_dbkm, (:dB, :km), (:linear, :m))
medium = Medium(0.15, 0.11, [-11.83e-27, 8.03e-41], alpha_npm, 835e-9)
linop = dispersion_operator(grid, medium)
# linop[i] = i*(β₂/2 * Δω² + β₃/6 * Δω³) - α/2
```

# Units

  - Input: `betas` in [sⁿ/m], `alpha` in [Nepers/m], `omega` in [rad/s]
  - Output: `linop` in [m⁻¹]

# Notes

    # grid.omega is the frequency detuning Δω = ω - ω₀

  - **Positive β₂**: Normal (positive) dispersion - longer wavelengths faster
  - **Negative β₂**: Anomalous (negative) dispersion - shorter wavelengths faster (solitons possible)
  - Higher-order terms (β₃, β₄, ...) cause pulse distortion and spectral asymmetry    # Initialize dispersion

# See Also

    # Taylor expansion of dispersion

  - [`apply_dispersion!`](@ref): Apply operator to frequency-domain field    # Note: betas are in SI units (s^n/m), omega is in rad/s    # grid.omega is the frequency detuning Δω = ω - ω₀
  - [`Medium`](@ref): Medium parameter structure    # betas[1] = beta2, betas[2] = beta3, etc. (beta0 and beta1 are omitted)
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

Apply dispersion operator to frequency-domain field in-place (optimized, no allocation).

# Mathematical Operation

```
Ã_out(ω) = Ã_in(ω) × exp(L̂(ω) × dz)
```

where L̂(ω) = iβ(ω) - α(ω)/2 is the linear operator.

# Arguments

  - `Aw::Vector{<:Complex}`: Frequency-domain field Ã(ω) [√W·s] (modified in-place)
  - `linop::Vector{<:Complex}`: Linear dispersion operator L̂(ω) [m⁻¹]
  - `dz::Real`: Propagation step [m]

# Returns

  - `nothing` (field is modified in-place)

# Performance

  - **Zero allocations**: Uses broadcasting for efficient in-place operation
  - Typical use: Called hundreds/thousands of times per simulation
  - Complexity: O(N) where N is grid size

# Example

```julia
linop = dispersion_operator(grid, medium)
Aw = ifft(pulse.At)
apply_dispersion!(Aw, linop, 0.001)  # Propagate 1 mm
At = fft(Aw)  # Transform back to time domain
```

# See Also

  - [`apply_dispersion`](@ref): Allocating version (returns new array)
  - [`dispersion_operator`](@ref): Construct linear operator
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
