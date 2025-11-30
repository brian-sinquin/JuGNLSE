# Integration tests for SSFM solver
using Test
using JuGNLSE

@testset "SSFM Solver" begin
    @testset "Basic Propagation" begin
        # Setup simple problem
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.01, 0.11, [-11.83e-27], 0.0, 835e-9)
        params = SimParams(medium=medium, N=2^10, n_saves=10, raman=false, shock=false)
        pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
        
        # Propagate
        results = solve(pulse, params, method=:ssfm, progress=false)
        
        @test results.z isa Vector
        @test results.At isa Matrix
        @test results.Aw isa Matrix
        @test size(results.At, 1) == grid.N
        @test size(results.At, 2) == params.n_saves
    end
    
    @testset "Energy Conservation (No Loss)" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.05, 0.11, [-11.83e-27], 0.0, 835e-9)
        params = SimParams(medium=medium, N=2^10, n_saves=20, raman=false, shock=false)
        pulse = sech_pulse(grid, 50e-15, 5000.0, 835e-9)
        
        results = solve(pulse, params, method=:ssfm, progress=false)
        
        # Calculate energy at each step (iterate over save points)
        energies = [pulse_energy(results.At[:,i], grid.dt) for i in 1:size(results.At, 2)]
        
        # Energy should be roughly conserved (within numerical error)
        @test all(abs.(energies .- energies[1]) ./ energies[1] .< 0.01)
    end
    
    @testset "Linear Dispersion Only" begin
        # Pure dispersion - should match analytical solution for Gaussian
        grid = create_grid(2^12, 20e-12, 835e-9)
        medium = Medium(0.001, 0.0, [-11.83e-27], 0.0, 835e-9)  # γ=0 (linear)
        params = SimParams(medium=medium, N=2^12, n_saves=5, raman=false, shock=false)
        
        T0 = 50e-15
        pulse = gaussian_pulse(grid, 2*sqrt(log(2))*T0, 1000.0, 835e-9)
        
        results = solve(pulse, params, method=:ssfm, progress=false)
        
        # Pulse should broaden
        I_initial = abs2.(results.At[:,1])
        I_final = abs2.(results.At[:,end])
        
        @test maximum(I_final) < maximum(I_initial)  # Peak decreases
    end
    
    @testset "Consistency Check" begin
        # Run twice with same parameters should give same result
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.01, 0.11, [-11.83e-27], 0.0, 835e-9)
        params = SimParams(medium=medium, N=2^10, n_saves=5, raman=false, shock=false)
        pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
        
        results1 = solve(pulse, params, method=:ssfm, progress=false)
        results2 = solve(pulse, params, method=:ssfm, progress=false)
        
        @test results1.At ≈ results2.At rtol=1e-10
    end
end
