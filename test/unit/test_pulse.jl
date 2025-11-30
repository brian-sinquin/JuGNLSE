# Unit tests for Pulse initialization and manipulation
using Test
using JuGNLSE
using FFTW

@testset "Pulse" begin
    # Setup common grid for tests
    grid = create_grid(2^12, 10e-12, 835e-9)
    
    @testset "Sech Pulse" begin
        FWHM = 50e-15
        P0 = 10000.0
        lambda0 = 835e-9
        
        pulse = sech_pulse(grid, FWHM, P0, lambda0)
        
        # Check structure
        @test pulse isa Pulse
        @test length(pulse.At) == grid.N
        @test length(pulse.Aw) == grid.N
        @test pulse.grid === grid
        
        # Check peak power
        @test peak_power(pulse.At) ≈ P0 rtol=1e-10
        
        # Check energy is finite and positive
        E = pulse_energy(pulse.At, grid.dt)
        @test E > 0
        @test isfinite(E)
        
        # Check pulse is centered (for time_offset=0)
        intensity = abs2.(pulse.At)
        peak_idx = argmax(intensity)
        center_idx = grid.N ÷ 2 + 1
        @test abs(peak_idx - center_idx) < 10  # Within a few points
    end
    
    @testset "Gaussian Pulse" begin
        FWHM = 50e-15
        P0 = 10000.0
        lambda0 = 835e-9
        
        pulse = gaussian_pulse(grid, FWHM, P0, lambda0)
        
        @test pulse isa Pulse
        @test peak_power(pulse.At) ≈ P0 rtol=1e-10
        
        E = pulse_energy(pulse.At, grid.dt)
        @test E > 0
        @test isfinite(E)
    end
    
    @testset "CW Pulse" begin
        P = 1000.0
        pulse = cw_pulse(grid, P)
        
        @test pulse isa Pulse
        
        # All time points should have same power
        intensity = abs2.(pulse.At)
        @test all(x -> isapprox(x, P, rtol=1e-10), intensity)
        
        # Energy should be P * total_time
        E = pulse_energy(pulse.At, grid.dt)
        @test E ≈ P * (grid.N * grid.dt) rtol=1e-10
    end
    
    @testset "Custom Pulse" begin
        # Create custom pulse (simple Gaussian)
        At_custom = ComplexF64.(@. exp(-grid.t^2 / (2*(50e-15)^2)))
        pulse = custom_pulse(grid, At_custom)
        
        @test pulse isa Pulse
        @test pulse.At ≈ At_custom
        @test length(pulse.Aw) == grid.N
    end
    
    @testset "Validation - Positive FWHM" begin
        @test_throws ArgumentError sech_pulse(grid, 0.0, 1000.0, 835e-9)
        @test_throws ArgumentError sech_pulse(grid, -50e-15, 1000.0, 835e-9)
        @test_throws ArgumentError gaussian_pulse(grid, 0.0, 1000.0, 835e-9)
        @test_throws ArgumentError gaussian_pulse(grid, -50e-15, 1000.0, 835e-9)
    end
    
    @testset "Validation - Non-negative Power" begin
        @test_throws ArgumentError sech_pulse(grid, 50e-15, -1000.0, 835e-9)
        @test_throws ArgumentError gaussian_pulse(grid, 50e-15, -1000.0, 835e-9)
        @test_throws ArgumentError cw_pulse(grid, -1000.0)
    end
    
    @testset "Validation - Array Size" begin
        wrong_size_At = zeros(ComplexF64, 100)
        @test_throws ArgumentError custom_pulse(grid, wrong_size_At)
    end
    
    @testset "Time Offset" begin
        FWHM = 50e-15
        P0 = 10000.0
        lambda0 = 835e-9
        offset = 1e-12  # 1 ps offset
        
        pulse = sech_pulse(grid, FWHM, P0, lambda0, time_offset=offset)
        
        # Peak should be shifted
        intensity = abs2.(pulse.At)
        peak_idx = argmax(intensity)
        peak_time = grid.t[peak_idx]
        @test peak_time ≈ offset rtol=1e-2  # Within 1%
    end
    
    @testset "Phase" begin
        FWHM = 50e-15
        P0 = 10000.0
        lambda0 = 835e-9
        phase = π/4
        
        pulse = sech_pulse(grid, FWHM, P0, lambda0, phase=phase)
        
        # Check phase at peak
        peak_idx = argmax(abs2.(pulse.At))
        measured_phase = angle(pulse.At[peak_idx])
        @test measured_phase ≈ phase rtol=1e-10
        
        # Intensity should be unchanged
        @test peak_power(pulse.At) ≈ P0 rtol=1e-10
    end
    
    @testset "Chirp" begin
        FWHM = 50e-15
        P0 = 10000.0
        lambda0 = 835e-9
        chirp = 1.0
        
        pulse_unchirped = sech_pulse(grid, FWHM, P0, lambda0)
        pulse_chirped = sech_pulse(grid, FWHM, P0, lambda0, chirp=chirp)
        
        # Intensity profiles should be same
        @test abs2.(pulse_unchirped.At) ≈ abs2.(pulse_chirped.At) rtol=1e-10
        
        # Spectral widths should differ
        spec_unchirped = abs2.(pulse_unchirped.Aw)
        spec_chirped = abs2.(pulse_chirped.Aw)
        
        E_unchirped = sum(spec_unchirped)
        E_chirped = sum(spec_chirped)
        
        # Energy should be conserved
        @test E_unchirped ≈ E_chirped rtol=1e-10
        
        # Chirped pulse should have broader spectrum
        # (This is a simplified check - more rigorous would compute actual bandwidth)
        @test maximum(spec_chirped) < maximum(spec_unchirped)
    end
    
    @testset "Sech vs Gaussian Shape" begin
        FWHM = 50e-15
        P0 = 10000.0
        lambda0 = 835e-9
        
        sech_p = sech_pulse(grid, FWHM, P0, lambda0)
        gauss_p = gaussian_pulse(grid, FWHM, P0, lambda0)
        
        # Both should have same peak power and FWHM
        @test peak_power(sech_p.At) ≈ P0 rtol=1e-10
        @test peak_power(gauss_p.At) ≈ P0 rtol=1e-10
        
        # Sech has more energy in wings (higher energy for same FWHM)
        E_sech = pulse_energy(sech_p.At, grid.dt)
        E_gauss = pulse_energy(gauss_p.At, grid.dt)
        @test E_sech > E_gauss
    end
    
    @testset "FFT Consistency" begin
        FWHM = 50e-15
        P0 = 10000.0
        lambda0 = 835e-9
        
        pulse = sech_pulse(grid, FWHM, P0, lambda0)
        
        # Aw should be ifft(At)
        Aw_check = ifft(pulse.At)
        @test pulse.Aw ≈ Aw_check rtol=1e-10
        
        # Roundtrip: At -> Aw -> At
        At_reconstructed = fft(pulse.Aw)
        @test pulse.At ≈ At_reconstructed rtol=1e-10
    end
    
    @testset "Energy Conservation" begin
        # Test FFT roundtrip preserves energy
        FWHM = 50e-15
        P0 = 10000.0
        lambda0 = 835e-9
        
        pulse = sech_pulse(grid, FWHM, P0, lambda0)
        
        # Aw = ifft(At), so fft(Aw) should give back At
        At_reconstructed = fft(pulse.Aw)
        @test pulse.At ≈ At_reconstructed rtol=1e-10
        
        # Energy in both representations
        E_t = pulse_energy(pulse.At, grid.dt)
        E_w = pulse_energy(At_reconstructed, grid.dt)
        @test E_t ≈ E_w rtol=1e-10
    end
    
    @testset "Different Pulse Widths" begin
        P0 = 10000.0
        lambda0 = 835e-9
        
        # Short pulse
        pulse_short = sech_pulse(grid, 10e-15, P0, lambda0)
        E_short = pulse_energy(pulse_short.At, grid.dt)
        
        # Medium pulse
        pulse_mid = sech_pulse(grid, 50e-15, P0, lambda0)
        E_mid = pulse_energy(pulse_mid.At, grid.dt)
        
        # Long pulse
        pulse_long = sech_pulse(grid, 200e-15, P0, lambda0)
        E_long = pulse_energy(pulse_long.At, grid.dt)
        
        # Longer pulses have more energy (same peak power)
        @test E_long > E_mid > E_short
        
        # All have same peak power
        @test peak_power(pulse_short.At) ≈ P0 rtol=1e-10
        @test peak_power(pulse_mid.At) ≈ P0 rtol=1e-10
        @test peak_power(pulse_long.At) ≈ P0 rtol=1e-10
    end
end
