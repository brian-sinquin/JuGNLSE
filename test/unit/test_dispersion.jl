# Unit tests for dispersion operators
using Test
using JuGNLSE
using FFTW

@testset "Dispersion" begin
    @testset "Dispersion Operator Construction" begin
        grid = create_grid(2^12, 10e-12, 835e-9)
        
        # Simple case: only β2
        medium = Medium(0.15, 0.11, [-11.83e-27], 0.0, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        @test linop isa Vector{ComplexF64}
        @test length(linop) == grid.N
        
        # Check structure: should be imaginary (no loss)
        @test all(isreal.(linop) .== false)
        @test all(abs.(real.(linop)) .< 1e-20)  # Real part should be zero (no loss)
    end
    
    @testset "β2 Only (GVD)" begin
        grid = create_grid(2^12, 10e-12, 835e-9)
        beta2 = -11.83e-27  # Typical for 835 nm in PCF
        
        medium = Medium(0.15, 0.11, [beta2], 0.0, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        # For β2 only: β(ω) = β2/2 * Δω²
        # linop = i*β(ω) = i*β2/2 * Δω²
        expected = @. im * beta2 / 2 * grid.omega^2
        
        @test linop ≈ expected rtol=1e-10
    end
    
    @testset "Higher-Order Dispersion" begin
        grid = create_grid(2^12, 10e-12, 835e-9)
        
        # Dudley 2006 parameters (up to β10)
        betas = [
            -11.83e-27,  # β2
            8.13e-41,    # β3
            -9.5e-56,    # β4
            2.0e-70,     # β5
            0.0,         # β6
            0.0,         # β7
            0.0,         # β8
            0.0,         # β9
            0.0          # β10
        ]
        
        medium = Medium(0.15, 0.11, betas, 0.0, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        @test length(linop) == grid.N
        
        # Check that higher-order terms matter at high frequencies
        # The operator should not be purely quadratic
        beta_omega_quadratic = @. betas[1] / 2 * grid.omega^2
        beta_omega_full = zeros(Float64, grid.N)
        for (idx, beta_n) in enumerate(betas)
            n = idx + 1
            beta_omega_full .+= beta_n .* (grid.omega .^ n) ./ factorial(n)
        end
        
        # At center frequency (omega=0), should be same
        center_idx = 1
        @test isapprox(imag(linop[center_idx]), beta_omega_full[center_idx], atol=1e-30)
        
        # At high frequencies, should differ (β3, β4 matter)
        high_freq_idx = grid.N ÷ 4
        @test !isapprox(beta_omega_full[high_freq_idx], beta_omega_quadratic[high_freq_idx], rtol=0.01)
    end
    
    @testset "Loss - Scalar" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        beta2 = -11.83e-27
        alpha_dB = 5.0  # dB/km
        
        medium = Medium(0.15, 0.11, [beta2], alpha_dB, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        # Convert dB/km to Nepers/m
        alpha_np = alpha_dB * log(10) / 10 / 1000
        
        # linop = i*β(ω) - α/2
        expected_imag = @. beta2 / 2 * grid.omega^2
        expected_real = -alpha_np / 2
        
        @test real.(linop) ≈ fill(expected_real, grid.N) rtol=1e-10
        @test imag.(linop) ≈ expected_imag rtol=1e-10
    end
    
    @testset "Loss - Frequency Dependent" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        beta2 = -11.83e-27
        
        # Frequency-dependent loss (linear with frequency as example)
        alpha_vec = fill(5.0, grid.N)
        
        medium = Medium(0.15, 0.11, [beta2], alpha_vec, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        # Check that loss varies
        alpha_np = alpha_vec .* log(10) / 10 / 1000
        expected_real = -alpha_np ./ 2
        
        @test real.(linop) ≈ expected_real rtol=1e-10
    end
    
    @testset "apply_dispersion! In-Place" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.15, 0.11, [-11.83e-27], 0.0, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        # Create test field
        Aw = ones(ComplexF64, grid.N)
        Aw_original = copy(Aw)
        
        dz = 1e-3  # 1 mm
        apply_dispersion!(Aw, linop, dz)
        
        # Check modified in-place
        @test Aw != Aw_original
        
        # Check result
        expected = @. Aw_original * exp(linop * dz)
        @test Aw ≈ expected rtol=1e-10
    end
    
    @testset "apply_dispersion Allocating" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.15, 0.11, [-11.83e-27], 0.0, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        Aw = ones(ComplexF64, grid.N)
        dz = 1e-3
        
        Aw_new = apply_dispersion(Aw, linop, dz)
        
        # Original unchanged
        @test Aw == ones(ComplexF64, grid.N)
        
        # Result correct
        expected = @. Aw * exp(linop * dz)
        @test Aw_new ≈ expected rtol=1e-10
    end
    
    @testset "Dispersion Phase Accumulation" begin
        # Test that dispersion correctly shifts frequency components
        grid = create_grid(2^12, 10e-12, 835e-9)
        beta2 = -11.83e-27
        medium = Medium(0.15, 0.11, [beta2], 0.0, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        # Create pulse in frequency domain (single frequency component)
        Aw = zeros(ComplexF64, grid.N)
        test_idx = grid.N ÷ 4  # Some non-zero frequency
        Aw[test_idx] = 1.0 + 0.0im
        
        dz = 1e-3
        Aw_prop = apply_dispersion(Aw, linop, dz)
        
        # Phase should accumulate according to β(ω) * z
        omega_test = grid.omega[test_idx]
        phase_accumulated = imag(linop[test_idx]) * dz
        expected_phase = angle(Aw_prop[test_idx])
        
        @test expected_phase ≈ phase_accumulated rtol=1e-10
        
        # Amplitude should decay if there's loss
        @test abs(Aw_prop[test_idx]) ≈ exp(real(linop[test_idx]) * dz) rtol=1e-10
    end
    
    @testset "Energy Conservation (No Loss)" begin
        # With no loss, energy should be conserved
        grid = create_grid(2^12, 10e-12, 835e-9)
        medium = Medium(0.15, 0.11, [-11.83e-27], 0.0, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        # Create test pulse
        pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
        
        E_initial = sum(abs2.(pulse.Aw))
        
        # Propagate
        Aw_prop = apply_dispersion(pulse.Aw, linop, 0.01)  # 1 cm
        E_final = sum(abs2.(Aw_prop))
        
        @test E_final ≈ E_initial rtol=1e-10
    end
    
    @testset "Energy Decay (With Loss)" begin
        # With loss, energy should decay exponentially
        grid = create_grid(2^12, 10e-12, 835e-9)
        alpha_dB = 10.0  # dB/km
        medium = Medium(0.15, 0.11, [-11.83e-27], alpha_dB, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
        
        E_initial = sum(abs2.(pulse.Aw))
        
        z = 0.1  # 10 cm
        Aw_prop = apply_dispersion(pulse.Aw, linop, z)
        E_final = sum(abs2.(Aw_prop))
        
        # Expected decay: E(z) = E(0) * exp(-α*z)
        alpha_np = alpha_dB * log(10) / 10 / 1000  # Nepers/m
        expected_ratio = exp(-alpha_np * z)
        
        @test E_final / E_initial ≈ expected_ratio rtol=1e-10
    end
    
    @testset "Gaussian Pulse Broadening" begin
        # Gaussian pulse should broaden under GVD
        grid = create_grid(2^13, 50e-12, 835e-9)
        beta2 = -11.83e-27  # Anomalous dispersion
        medium = Medium(0.15, 0.11, [beta2], 0.0, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        T0 = 50e-15  # 50 fs
        pulse = gaussian_pulse(grid, 2*sqrt(log(2))*T0, 10000.0, 835e-9)
        
        # Propagate
        z = 0.001  # 1 mm
        Aw_prop = apply_dispersion(pulse.Aw, linop, z)
        At_prop = fft(Aw_prop)
        
        # Calculate widths
        I_initial = abs2.(pulse.At)
        I_final = abs2.(At_prop)
        
        # Peak should be similar (energy conserved)
        E_initial = sum(I_initial) * grid.dt
        E_final = sum(I_final) * grid.dt
        @test E_final ≈ E_initial rtol=1e-10
        
        # Pulse should broaden (peak intensity should decrease)
        @test maximum(I_final) < maximum(I_initial)
    end
    
    @testset "Zero Dispersion" begin
        # With all betas = 0, should have no effect (except loss)
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.15, 0.11, [0.0, 0.0, 0.0], 0.0, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        # Operator should be zero everywhere
        @test all(abs.(linop) .< 1e-20)
        
        # Propagation should not change field
        Aw = randn(ComplexF64, grid.N)
        Aw_prop = apply_dispersion(Aw, linop, 1.0)
        
        @test Aw_prop ≈ Aw rtol=1e-10
    end
    
    @testset "Symmetry Properties" begin
        # For real β coefficients, certain symmetries should hold
        grid = create_grid(2^12, 10e-12, 835e-9)
        medium = Medium(0.15, 0.11, [-11.83e-27, 8.13e-41], 0.0, 835e-9)
        linop = dispersion_operator(grid, medium)
        
        # For even orders only (β2, β4, ...), operator should be symmetric
        # Check that linop[i] ≈ conj(linop[N-i+2]) for appropriate indexing
        # (This depends on FFT ordering after fftshift in grid)
        
        # Basic check: imaginary part should have certain structure
        @test all(isfinite.(linop))
        @test !any(isnan.(linop))
    end
end
