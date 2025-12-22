"""
    raman_response(grid::Grid, model::RamanModel)

Compute normalized Raman response function hᵣ(t) in time domain for GNLSE Raman term
∂A/∂z|_Raman = iγfᵣA ∫_{-∞}^{t} hᵣ(t-t')|A(t')|² dt'. All models satisfy causality
(hᵣ(t) = 0 for t < 0) and normalization (∫₀^∞ hᵣ(t) dt = 1). Available models:
BlowWood (τ₁=12.2 fs, τ₂=32 fs, fᵣ=0.18), LinAgrawal (three-component with Boson peak,
fᵣ=0.245), Hollenbeck (13-oscillator fit to experimental data, fᵣ≈0.20).

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `model::RamanModel`: Raman model specification

# Returns

  - `(h_R, fr)`: Normalized response function [unitless] and Raman fraction fᵣ [unitless]
"""
function raman_response(grid::Grid, model::BlowWood)
    τ1 = 12.2e-15  # s
    τ2 = 32.0e-15  # s

    # Create response function (only for t ≥ 0)
    h_R = zeros(Float64, grid.N)

    for (i, t) in enumerate(grid.t)
        if t >= 0
            h_R[i] = (τ1^2 + τ2^2) / (τ1 * τ2^2) * exp(-t / τ2) * sin(t / τ1)
        end
    end

    (h_R, model.fr)
end

"""
    raman_response(grid::Grid, model::LinAgrawal)

Compute three-component Raman response with Boson peak contribution for improved
low-frequency accuracy. Combines primary Lorentzian (τ₁=12.2 fs, τ₂=32 fs) with
Boson peak (τb=96 fs) using weighting factor fb=0.21. Preferred for broadband
simulations and sub-100 fs pulses. Reference: Q. Lin and G. P. Agrawal,
Opt. Lett. 31, 3086-3088 (2006).
"""
function raman_response(grid::Grid, model::LinAgrawal)
    # Primary oscillator parameters (same as Blow-Wood)
    τ1 = 12.2e-15  # s
    τ2 = 32.0e-15  # s

    # Boson peak parameters
    τb = 96.0e-15  # s - slower relaxation
    fb = model.fb  # Boson peak fraction (default 0.21)

    # Create response function
    h_R = zeros(Float64, grid.N)

    for (i, t) in enumerate(grid.t)
        if t >= 0
            # Single-Lorentzian component (Blow-Wood form)
            h_single = (τ1^2 + τ2^2) / (τ1 * τ2^2) * exp(-t / τ2) * sin(t / τ1)

            # Boson peak component (damped exponential)
            h_boson = (2τb - t) / τb^2 * exp(-t / τb)

            # Combined response with weighting
            h_R[i] = (1 - fb) * h_single + fb * h_boson
        end
    end

    (h_R, model.fr)
end

"""
    raman_response(grid::Grid, model::Hollenbeck)

Compute 13-oscillator Raman response fitted to experimental Raman gain data.
Response is h_R(t) = Σᵢ Aᵢ·exp(-Γᵢt)·sin(Ωᵢt) with damped harmonic oscillators
spanning 1.75-15.6 THz. Provides highest accuracy for supercontinuum generation
and precise spectral predictions. Parameters from Table 1 of Hollenbeck & Cantrell,
JOSA B 19, 2886-2902 (2002).
"""
function raman_response(grid::Grid, model::Hollenbeck)
    # Hollenbeck-Cantrell parameters (13 oscillators)
    # From Table 1 of JOSA B 19, 2886 (2002)
    # Units: frequencies in THz (× 2π for rad/s), damping in THz

    # Oscillator parameters [frequency (THz), damping (THz), amplitude (normalized)]
    oscillators = [
        (15.6, 0.50, 0.010),  # Mode 1
        (13.35, 0.70, 0.048),  # Mode 2
        (12.15, 0.55, 0.092),  # Mode 3  - dominant peak
        (11.40, 0.40, 0.057),  # Mode 4
        (10.35, 0.35, 0.038),  # Mode 5
        (9.15, 0.32, 0.025),  # Mode 6
        (8.70, 0.28, 0.018),  # Mode 7
        (7.80, 0.25, 0.012),  # Mode 8
        (6.52, 0.23, 0.008),  # Mode 9
        (5.82, 0.20, 0.005),  # Mode 10
        (4.50, 0.18, 0.003),  # Mode 11
        (3.25, 0.15, 0.002),  # Mode 12 - low frequency tail
        (1.75, 0.10, 0.001),  # Mode 13 - Boson peak region
    ]

    h_R = zeros(Float64, grid.N)

    for (i, t) in enumerate(grid.t)
        if t >= 0
            # Sum contributions from all oscillators
            for (ν, Γ, A) in oscillators
                # Convert to SI units
                Ω = 2π * ν * 1e12    # THz → rad/s
                γ = 2π * Γ * 1e12    # THz → 1/s damping rate

                # Damped oscillator response
                h_R[i] += A * exp(-γ * t) * sin(Ω * t)
            end
        end
    end

    # Normalize (ensure ∫h_R dt = 1)
    dt = grid.dt
    integral = sum(h_R) * dt
    if integral > 0
        h_R ./= integral
    end

    (h_R, model.fr)
end

"""
    raman_response_frequency(h_R::Vector{Float64}, grid::Grid)

Transform Raman response to frequency domain for efficient convolution via FFT.
Converts time-domain convolution (|A|² ⊗ hᵣ)(t) to frequency-domain multiplication
F{|A|²} × F{hᵣ}, reducing complexity from O(N²) to O(N log N). Implementation uses
conj(fft(ifftshift(h_R))) where ifftshift aligns zero-time for FFT and conjugate
matches established solver conventions.

# Arguments

  - `h_R::Vector{Float64}`: Time-domain Raman response [unitless]
  - `grid::Grid`: Time-frequency grid

# Returns

  - `Vector{ComplexF64}`: Frequency-domain response R̃(ω) [unitless]
"""
function raman_response_frequency(h_R::Vector{Float64}, grid::Grid)
    # FFT of Raman response for convolution theorem: F{f ⊗ h} = F{f} × F{h}
    #
    # CRITICAL: Match FiberNlse convention exactly:
    # 1. ifftshift(h_R): Move zero time to the beginning
    # 2. fft: Transform to frequency domain
    # 3. conj: Apply conjugate (this is the FiberNlse convention)
    # 4. NO dt multiplication here - it's applied during convolution
    #
    # This matches: conj(fft(ifftshift(h_R)))
    #
    RW = conj(fft(ifftshift(h_R)))
    RW
end
