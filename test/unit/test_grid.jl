# Unit tests for Grid creation and properties
using Test
using JuGNLSE
using FFTW

@testset "Grid" begin
    @testset "Basic Construction" begin
        # Standard parameters
        N = 2^10
        time_window = 10e-12
        lambda0 = 835e-9
        
        grid = create_grid(N, time_window, lambda0)
        
        @test grid.N == N
        @test length(grid.t) == N
        @test length(grid.omega) == N
        @test grid.dt ≈ time_window / N
        @test grid.domega ≈ 2π / time_window
    end
    
    @testset "Time Array Properties" begin
        N = 2^12
        time_window = 10e-12
        lambda0 = 835e-9
        
        grid = create_grid(N, time_window, lambda0)
        
        # Time should be centered around 0
        @test grid.t[1] ≈ -time_window/2
        @test grid.t[end] ≈ time_window/2 - grid.dt
        
        # Check uniform spacing
        dt_check = diff(grid.t)
        @test all(x -> isapprox(x, grid.dt, rtol=1e-10), dt_check)
        
        # Check symmetry
        @test isapprox(grid.t[end], -grid.t[1] - grid.dt, rtol=1e-10)
    end
    
    @testset "Frequency Array Properties" begin
        N = 2^12
        time_window = 10e-12
        lambda0 = 835e-9
        
        grid = create_grid(N, time_window, lambda0)
        
        # Frequency should be in FFT order after fftshift
        # [0, dω, 2dω, ..., ωmax, -ωmax, ..., -dω]
        # Note: first element after fftshift from negative frequencies
        
        # Check Nyquist frequency
        omega_max = π / grid.dt
        @test maximum(abs.(grid.omega)) ≈ omega_max rtol=1e-10
        
        # Check uniform spacing
        domega_check = abs(grid.omega[2] - grid.omega[1])
        @test domega_check ≈ grid.domega rtol=1e-10
    end
    
    @testset "FFT Convention" begin
        N = 2^10
        time_window = 10e-12
        lambda0 = 835e-9
        
        grid = create_grid(N, time_window, lambda0)
        
        # Test that FFT works correctly with this grid
        # Create a test pulse (Gaussian)
        t0 = 50e-15
        At = @. exp(-grid.t^2 / (2*t0^2))
        
        # Transform to frequency domain
        Aw = fft(At)
        
        # Transform back
        At_reconstructed = ifft(Aw)
        
        @test isapprox(At, At_reconstructed, rtol=1e-10)
    end
    
    @testset "Validation - Positive N" begin
        @test_throws ArgumentError create_grid(0, 10e-12, 835e-9)
        @test_throws ArgumentError create_grid(-10, 10e-12, 835e-9)
    end
    
    @testset "Validation - Positive Time Window" begin
        @test_throws ArgumentError create_grid(2^10, 0.0, 835e-9)
        @test_throws ArgumentError create_grid(2^10, -10e-12, 835e-9)
    end
    
    @testset "Validation - Positive Wavelength" begin
        @test_throws ArgumentError create_grid(2^10, 10e-12, 0.0)
        @test_throws ArgumentError create_grid(2^10, 10e-12, -835e-9)
    end
    
    @testset "N Power of 2 Warning" begin
        # Power of 2 - should work
        grid = create_grid(2^12, 10e-12, 835e-9)
        @test grid.N == 2^12
        
        # Non-power of 2 - should warn but work
        grid2 = create_grid(1000, 10e-12, 835e-9)
        @test grid2.N == 1000
    end
    
    @testset "Different Resolutions" begin
        lambda0 = 835e-9
        time_window = 10e-12
        
        # Low resolution
        grid_low = create_grid(2^8, time_window, lambda0)
        @test grid_low.N == 256
        @test grid_low.dt ≈ time_window / 256
        
        # Medium resolution
        grid_mid = create_grid(2^12, time_window, lambda0)
        @test grid_mid.N == 4096
        @test grid_mid.dt ≈ time_window / 4096
        
        # High resolution
        grid_high = create_grid(2^14, time_window, lambda0)
        @test grid_high.N == 16384
        @test grid_high.dt ≈ time_window / 16384
        
        # Higher N -> smaller dt
        @test grid_high.dt < grid_mid.dt < grid_low.dt
    end
    
    @testset "Different Time Windows" begin
        N = 2^12
        lambda0 = 835e-9
        
        # Short window (ultrafast)
        grid_short = create_grid(N, 1e-12, lambda0)
        @test grid_short.dt ≈ 1e-12 / N
        
        # Medium window
        grid_mid = create_grid(N, 10e-12, lambda0)
        @test grid_mid.dt ≈ 10e-12 / N
        
        # Long window
        grid_long = create_grid(N, 100e-12, lambda0)
        @test grid_long.dt ≈ 100e-12 / N
        
        # Longer window -> larger dt
        @test grid_long.dt > grid_mid.dt > grid_short.dt
    end
    
    @testset "Parseval's Theorem" begin
        # Energy should be conserved between time and frequency domains
        N = 2^12
        grid = create_grid(N, 10e-12, 835e-9)
        
        # Create test pulse
        t0 = 50e-15
        At = @. exp(-grid.t^2 / (2*t0^2))
        
        # FFT and back should preserve signal
        Aw = fft(At)
        At_back = ifft(Aw)
        
        # Test roundtrip accuracy
        @test isapprox(At, At_back, rtol=1e-10)
        
        # Test Parseval: sum|At|² = (1/N) * sum|Aw|²
        # This is the FFT normalization in Julia
        @test isapprox(sum(abs2.(At)), sum(abs2.(Aw))/N, rtol=1e-10)
    end
    
    @testset "Grid from Medium" begin
        medium = Medium(0.15, 0.11, [0.0, 0.0, -11.83e-27], 0.0, 835e-9)
        
        # With explicit time window
        grid1 = create_grid_from_medium(2^12, medium, time_window=10e-12)
        @test grid1.N == 2^12
        @test grid1.dt ≈ 10e-12 / 2^12
        
        # With default time window
        grid2 = create_grid_from_medium(2^12, medium)
        @test grid2.N == 2^12
        @test grid2.dt ≈ 20e-12 / 2^12  # Default is 20ps
    end
end
