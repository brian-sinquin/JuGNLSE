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
- Loss term: α converted from dB/km to Nepers/m: α[Np/m] = α[dB/km] × ln(10)/(10×1000)

# Arguments
- `grid::Grid`: Time-frequency grid containing Δω array
- `medium::Medium`: Medium parameters with beta coefficients and loss

# Returns
- `Vector{ComplexF64}`: Linear operator L̂(ω) = iβ(ω) - α(ω)/2

# Example
```julia
grid = create_grid(2^12, 10e-12, 835e-9)
medium = Medium(0.15, 0.11, [-11.83e-27, 8.03e-41], 0.0, 835e-9)
linop = dispersion_operator(grid, medium)
# linop[i] = i*(β₂/2 * Δω² + β₃/6 * Δω³) - α/2
```

# Units
- Input: `betas` in [sⁿ/m], `alpha` in [dB/km], `omega` in [rad/s]
- Output: `linop` in [m⁻¹]

# Notes
- **Positive β₂**: Normal (positive) dispersion - longer wavelengths faster
- **Negative β₂**: Anomalous (negative) dispersion - shorter wavelengths faster (solitons possible)
- Higher-order terms (β₃, β₄, ...) cause pulse distortion and spectral asymmetry

# See Also
- [`apply_dispersion!`](@ref): Apply operator to frequency-domain field
- [`Medium`](@ref): Medium parameter structure
"""
function dispersion_operator(grid::Grid, medium::Medium)
    # Center frequency (not used - grid.omega is already detuning from center)
    omega0 = 2π * 3e8 / medium.lambda0
    
    # grid.omega is already the frequency detuning Δω = ω - ω₀
    domega = grid.omega
    
    # Initialize dispersion
    beta_omega = zeros(Float64, grid.N)
    
    # Taylor expansion of dispersion
    # Note: betas are in SI units (s^n/m), omega is in rad/s
    # betas[1] = beta2, betas[2] = beta3, etc. (beta0 and beta1 are omitted)
    for (idx, beta_n) in enumerate(medium.betas)
        n = idx + 1  # betas[1] is beta2, so n = 2, 3, 4, ...
        # Higher order terms: βₙ/n! * Δωⁿ
        # Standard convention: β(ω) = Σ (βₙ/n!) Δω^n where βₙ = d^nβ/dω^n
        beta_omega .+= beta_n .* (domega .^ n) ./ factorial(n)
    end
    
    # Add loss term (negative imaginary part)
    # Convert dB/km to 1/m
    # Standard GNLSE: ∂Ã/∂z = +iβ(ω)Ã - (α/2)Ã + nonlinear terms
    if medium.alpha isa Real
        alpha_np = medium.alpha * log(10) / 10 / 1000  # Nepers per meter
        linop = im .* beta_omega .- alpha_np / 2
    else
        # Frequency-dependent loss
        alpha_np = medium.alpha .* log(10) / 10 / 1000
        linop = im .* beta_omega .- alpha_np ./ 2
    end
    
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
