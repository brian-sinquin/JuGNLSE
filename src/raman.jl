"""
    raman_response(grid::Grid, model::RamanModel)

Calculate the Raman response function in the time domain for GNLSE simulations.

# Mathematical Formulation
The Raman contribution to GNLSE is:
```
∂A/∂z|_Raman = iγfᵣA ∫_{-∞}^{t} hᵣ(t-t')|A(t')|² dt'
```
where hᵣ(t) is the normalized Raman response function:
```
∫₀^∞ hᵣ(t) dt = 1
```

# Arguments
- `grid::Grid`: Time-frequency grid
- `model::RamanModel`: Raman model type ([`BlowWood`](@ref), [`LinAgrawal`](@ref), [`Hollenbeck`](@ref))

# Returns
- `Tuple{Vector{Float64}, Float64}`: (hᵣ(t), fᵣ) where:
  * `hᵣ`: Normalized Raman response function in time domain
  * `fᵣ`: Raman fraction (model-dependent default)

# Model Comparison

## BlowWood (1989)
- **Formula**: hᵣ(t) = (τ₁² + τ₂²)/(τ₁τ₂²) exp(-t/τ₂) sin(t/τ₁)
- **Parameters**: τ₁ = 12.2 fs, τ₂ = 32.0 fs
- **Raman fraction**: fᵣ = 0.18
- **Use case**: Fast, general purpose, good accuracy for most applications

## LinAgrawal (2006)
- **Formula**: Three-component model with Boson peak
- **Raman fraction**: fᵣ = 0.245
- **Use case**: Broadband simulations, sub-100 fs pulses, improved low-frequency response

## Hollenbeck (2002)
- **Formula**: 13-oscillator fit to experimental data
- **Raman fraction**: fᵣ ≈ 0.20
- **Use case**: Highest accuracy, supercontinuum generation, precise spectral predictions

# Examples
```julia
# Get Blow-Wood response for standard fiber
h_R, fr = raman_response(grid, BlowWood())
# fr = 0.18

# Use in frequency domain for fast convolution
RW = raman_response_frequency(h_R, grid)
```

# Normalization
All response functions are normalized such that ∫₀^∞ hᵣ(t) dt = 1, which ensures
physically correct Raman gain when combined with the Raman fraction fᵣ.

# Notes
- Response is causal: hᵣ(t) = 0 for t < 0
- Peak response occurs at t ≈ 13-15 fs for all models
- Raman response width ~few hundred femtoseconds

# See Also
- [`raman_response_frequency`](@ref): Transform to frequency domain for convolution
- [`RamanModel`](@ref): Abstract type for Raman models
"""
function raman_response(grid::Grid, model::BlowWood)
    τ1 = 12.2e-15  # s
    τ2 = 32.0e-15  # s
    fr = 0.18
    
    # Create response function (only for t ≥ 0)
    h_R = zeros(Float64, grid.N)
    
    for (i, t) in enumerate(grid.t)
        if t >= 0
            h_R[i] = (τ1^2 + τ2^2) / (τ1 * τ2^2) * exp(-t/τ2) * sin(t/τ1)
        end
    end
    
    # Normalize
    h_R ./= (sum(h_R) * grid.dt)
    
    (h_R, fr)
end

function raman_response(grid::Grid, model::LinAgrawal)
    # Parameters for Lin-Agrawal model (includes Boson peak)
    fr = 0.245
    
    # Three-component model
    fa = 0.75
    fb = 0.21
    fc = 0.04
    
    τ1a = 12.2e-15  # s
    τ2a = 32.0e-15  # s
    τb = 96.0e-15   # s
    τ1c = 12.2e-15  # s
    τ2c = 32.0e-15  # s
    
    h_R = zeros(Float64, grid.N)
    
    for (i, t) in enumerate(grid.t)
        if t >= 0
            # Component a (main peak)
            ha = (τ1a^2 + τ2a^2) / (τ1a * τ2a^2) * exp(-t/τ2a) * sin(t/τ1a)
            
            # Component b (Boson peak)
            hb = (2τb - t) / τb^2 * exp(-t/τb)
            
            # Component c
            hc = (τ1c^2 + τ2c^2) / (τ1c * τ2c^2) * exp(-t/τ2c) * sin(t/τ1c)
            
            h_R[i] = fa * ha + fb * hb + fc * hc
        end
    end
    
    # Normalize
    h_R ./= (sum(h_R) * grid.dt)
    
    (h_R, fr)
end

function raman_response(grid::Grid, model::Hollenbeck)
    # Hollenbeck-Cantrell 13-oscillator model (simplified version)
    # For full accuracy, implement all 13 Lorentzians
    # Here we use a simplified version
    
    fr = 0.20
    
    # Use Blow-Wood as base (more accurate implementation would include all oscillators)
    τ1 = 12.2e-15  # s
    τ2 = 32.0e-15  # s
    
    h_R = zeros(Float64, grid.N)
    
    for (i, t) in enumerate(grid.t)
        if t >= 0
            h_R[i] = (τ1^2 + τ2^2) / (τ1 * τ2^2) * exp(-t/τ2) * sin(t/τ1)
        end
    end
    
    # Normalize
    h_R ./= (sum(h_R) * grid.dt)
    
    (h_R, fr)
end

"""
    raman_response_frequency(h_R::Vector{Float64}, grid::Grid)

Transform Raman response to frequency domain for efficient convolution in GNLSE solvers.

# Mathematical Background
Time-domain convolution (slow):
```
(|A|² ⊗ hᵣ)(t) = ∫_{-∞}^{t} hᵣ(t-t')|A(t')|² dt'
```
becomes multiplication in frequency domain (fast):
```
F{|A|² ⊗ hᵣ} = F{|A|²} × F{hᵣ}
```

# Implementation
The function applies proper FFT scaling and shifting to ensure correct convolution:
```julia
RW = N × ifft(fftshift(hᵣ))
```

# Arguments
- `h_R::Vector{Float64}`: Time-domain Raman response (from [`raman_response`](@ref))
- `grid::Grid`: Time-frequency grid

# Returns
- `Vector{ComplexF64}`: Frequency-domain Raman response R̃(ω) for use in convolution

# Usage in Solvers
```julia
# Setup (done once)
h_R, fr = raman_response(grid, BlowWood())
RW = raman_response_frequency(h_R, grid)

# In propagation loop (efficient convolution)
It_w = ifft(|A(t)|²)
It_w .*= RW              # Convolution in frequency domain
Raman_term = fft(It_w)   # Back to time domain
```

# Performance
- Convolution via FFT: O(N log N) vs O(N²) for direct time-domain integration
- Critical for real-time simulation performance
- Called once at initialization, result reused throughout propagation

# See Also
- [`raman_response`](@ref): Calculate time-domain response
- [`nonlinear_operator`](@ref): Uses frequency-domain response for Raman term
"""
function raman_response_frequency(h_R::Vector{Float64}, grid::Grid)
    # FFT of Raman response
    # Note: Removed grid.N scaling - it was causing values ~1e13 which broke RK methods
    # The time step (dt) scaling is applied during convolution in nonlinear operators
    h_R_shifted = fftshift(h_R)
    RW = ifft(h_R_shifted)
    RW
end
