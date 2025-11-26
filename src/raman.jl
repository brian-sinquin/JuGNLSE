"""
    raman_response(grid::Grid, model::RamanModel)

Calculate Raman response function in time domain.

# Arguments
- `grid::Grid`: Time-frequency grid
- `model::RamanModel`: Raman model (BlowWood, LinAgrawal, or Hollenbeck)

# Returns
- `Tuple{Vector{Float64}, Float64}`: (h_R, fr) where h_R is the response function and fr is the Raman fraction
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

Transform Raman response to frequency domain for efficient convolution.

# Arguments
- `h_R::Vector{Float64}`: Time-domain Raman response
- `grid::Grid`: Time-frequency grid

# Returns
- `Vector{ComplexF64}`: Frequency-domain Raman response
"""
function raman_response_frequency(h_R::Vector{Float64}, grid::Grid)
    # FFT of Raman response, properly scaled
    h_R_shifted = fftshift(h_R)
    RW = grid.N .* ifft(h_R_shifted)
    RW
end
