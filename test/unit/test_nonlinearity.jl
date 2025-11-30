# Unit tests for nonlinear operators
using Test
using JuGNLSE
using FFTW

@testset "Nonlinearity" begin
    grid = create_grid(2^10, 10e-12, 835e-9)
    
    @testset "Pure Kerr Effect" begin
        # Simple Kerr: iγ|A|²A
        medium = Medium(0.15, 0.11, [0.0], 0.0, 835e-9)
        params = SimParams(medium=medium, raman=false, shock=false)
        
        # Create test pulse
        pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
        
        # Calculate nonlinear operator
        nonlin = nonlinear_operator(pulse.At, params, grid)
        
        @test length(nonlin) == grid.N
        @test all(isfinite.(nonlin))
        
        # For Kerr: N = iγ|A|²
        # Sign check: imaginary part should be positive where intensity is high
        peak_idx = argmax(abs2.(pulse.At))
        @test imag(nonlin[peak_idx]) > 0  # Positive imaginary (SPM)
    end
    
    @testset "Kerr + Raman" begin
        medium = Medium(0.15, 0.11, [0.0], 0.0, 835e-9)
        params = SimParams(medium=medium, raman=true, shock=false, raman_model=BlowWood())
        
        pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
        
        # Need Raman response
        h_R, fr = raman_response(grid, params.raman_model)
        RW = raman_response_frequency(h_R, grid)
        
        nonlin = nonlinear_operator(pulse.At, params, grid, RW)
        
        @test length(nonlin) == grid.N
        @test all(isfinite.(nonlin))
    end
    
    @testset "Kerr + Shock" begin
        medium = Medium(0.15, 0.11, [0.0], 0.0, 835e-9)
        params = SimParams(medium=medium, raman=false, shock=true)
        
        pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
        
        nonlin = nonlinear_operator(pulse.At, params, grid)
        
        @test length(nonlin) == grid.N
        @test all(isfinite.(nonlin))
    end
    
    @testset "Full Nonlinearity (Kerr + Raman + Shock)" begin
        medium = Medium(0.15, 0.11, [0.0], 0.0, 835e-9)
        params = SimParams(medium=medium, raman=true, shock=true)
        
        pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
        
        h_R, fr = raman_response(grid, params.raman_model)
        RW = raman_response_frequency(h_R, grid)
        
        nonlin = nonlinear_operator(pulse.At, params, grid, RW)
        
        @test length(nonlin) == grid.N
        @test all(isfinite.(nonlin))
    end
    
    @testset "Scaling with Power" begin
        # Nonlinearity should scale linearly with gamma
        medium1 = Medium(0.15, 0.1, [0.0], 0.0, 835e-9)
        medium2 = Medium(0.15, 0.2, [0.0], 0.0, 835e-9)
        
        params1 = SimParams(medium=medium1, raman=false, shock=false)
        params2 = SimParams(medium=medium2, raman=false, shock=false)
        
        pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
        
        nonlin1 = nonlinear_operator(pulse.At, params1, grid)
        nonlin2 = nonlinear_operator(pulse.At, params2, grid)
        
        # Nonlinearity should scale with gamma
        @test nonlin2 ≈ 2 .* nonlin1 rtol=1e-10
    end
    
    @testset "Zero at Zero Field" begin
        # Nonlinearity should be zero for zero field
        medium = Medium(0.15, 0.11, [0.0], 0.0, 835e-9)
        params = SimParams(medium=medium, raman=false, shock=false)
        
        At_zero = zeros(ComplexF64, grid.N)
        nonlin = nonlinear_operator(At_zero, params, grid)
        
        @test all(abs.(nonlin) .< 1e-20)
    end
end
