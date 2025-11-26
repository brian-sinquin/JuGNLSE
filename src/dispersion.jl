"""
    dispersion_operator(grid::Grid, medium::Medium)

Construct the linear dispersion operator in the frequency domain.

Uses Taylor expansion: β(ω) = Σ βₙ/n! * (ω - ω₀)ⁿ

# Arguments
- `grid::Grid`: Time-frequency grid
- `medium::Medium`: Medium parameters containing beta coefficients

# Returns
- `Vector{ComplexF64}`: Linear operator in frequency domain
"""
function dispersion_operator(grid::Grid, medium::Medium)
    # Center frequency
    omega0 = 2π * 3e8 / medium.lambda0
    
    # Frequency detuning
    domega = grid.omega
    
    # Initialize dispersion
    beta_omega = zeros(Float64, grid.N)
    
    # Taylor expansion of dispersion
    # Note: betas are in SI units (s^n/m), omega is in rad/s
    for (idx, beta_n) in enumerate(medium.betas)
        n = idx - 1  # Convert to dispersion order (0, 1, 2, ...)
        if n == 0
            # β₀ term (constant phase, typically omitted)
            # beta_omega .+= beta_n
        elseif n == 1
            # β₁ term (group delay, usually omitted in moving frame)
            # beta_omega .+= beta_n .* domega
        else
            # Higher order terms: βₙ/n! * Δωⁿ
            # Standard convention: β(ω) = Σ (βₙ/n!) Δω^n where βₙ = d^nβ/dω^n
            beta_omega .+= beta_n .* (domega .^ n) ./ factorial(n)
        end
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

Apply dispersion operator to frequency-domain field in-place.

# Arguments
- `Aw::Vector{<:Complex}`: Frequency-domain field (modified in-place)
- `linop::Vector{<:Complex}`: Linear dispersion operator
- `dz::Real`: Propagation step [m]
"""
function apply_dispersion!(Aw::Vector{<:Complex}, linop::Vector{<:Complex}, dz::Real)
    @. Aw *= exp(linop * dz)
    nothing
end

"""
    apply_dispersion(Aw::Vector{<:Complex}, linop::Vector{<:Complex}, dz::Real)

Apply dispersion operator to frequency-domain field (allocating version).

# Arguments
- `Aw::Vector{<:Complex}`: Frequency-domain field
- `linop::Vector{<:Complex}`: Linear dispersion operator
- `dz::Real`: Propagation step [m]

# Returns
- `Vector{ComplexF64}`: Propagated frequency-domain field
"""
function apply_dispersion(Aw::Vector{<:Complex}, linop::Vector{<:Complex}, dz::Real)
    @. Aw * exp(linop * dz)
end
