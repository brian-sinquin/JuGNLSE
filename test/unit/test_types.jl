# Unit tests for core types (Medium, Grid, Pulse, SimParams)
using Test
using JuGNLSE

@testset "Core Types" begin
    @testset "Medium" begin
        @testset "Basic Construction" begin
            # Scalar gamma (most common case)
            medium = Medium(0.15, 0.11, [0.0, 0.0, -11.83e-27], 0.0, 835e-9)
            @test medium.length == 0.15
            @test medium.gamma == 0.11
            @test length(medium.betas) == 3
            @test medium.alpha == 0.0
            @test medium.lambda0 == 835e-9
            @test medium.scaling === nothing
        end
        
        @testset "Validation - Positive Length" begin
            @test_throws ArgumentError Medium(-0.1, 0.11, [0.0], 0.0, 835e-9)
            @test_throws ArgumentError Medium(0.0, 0.11, [0.0], 0.0, 835e-9)
        end
        
        @testset "Validation - Non-negative Gamma" begin
            @test_throws ArgumentError Medium(0.15, -0.11, [0.0], 0.0, 835e-9)
        end
        
        @testset "Validation - Positive Wavelength" begin
            @test_throws ArgumentError Medium(0.15, 0.11, [0.0], 0.0, -835e-9)
            @test_throws ArgumentError Medium(0.15, 0.11, [0.0], 0.0, 0.0)
        end
        
        @testset "Frequency-Dependent Gamma" begin
            # Vector gamma (M-GNLSE)
            gamma_vec = fill(0.11, 100)
            scaling_vec = ones(100)
            
            # With scaling (correct usage)
            medium = Medium(0.15, gamma_vec, [0.0], 0.0, 835e-9, scaling_vec)
            @test medium.gamma isa Vector
            @test length(medium.gamma) == 100
            @test medium.scaling !== nothing
            @test length(medium.scaling) == 100
            
            # Without scaling (should warn but work)
            medium2 = Medium(0.15, gamma_vec, [0.0], 0.0, 835e-9)
            @test medium2.gamma isa Vector
            @test medium2.scaling === nothing
        end
        
        @testset "Validation - Scaling Factor" begin
            gamma_vec = fill(0.11, 100)
            scaling_vec = ones(100)
            
            # Scaling with scalar gamma should fail
            @test_throws ArgumentError Medium(0.15, 0.11, [0.0], 0.0, 835e-9, scaling_vec)
            
            # Mismatched lengths should fail
            wrong_scaling = ones(50)
            @test_throws ArgumentError Medium(0.15, gamma_vec, [0.0], 0.0, 835e-9, wrong_scaling)
        end
        
        @testset "Frequency-Dependent Loss" begin
            # Scalar loss
            medium1 = Medium(0.15, 0.11, [0.0], 5.0, 835e-9)
            @test medium1.alpha == 5.0
            
            # Vector loss
            alpha_vec = fill(5.0, 100)
            medium2 = Medium(0.15, 0.11, [0.0], alpha_vec, 835e-9)
            @test medium2.alpha isa Vector
            @test length(medium2.alpha) == 100
        end
        
        @testset "High-Order Dispersion" begin
            # Up to β₁₀ (Dudley 2006 uses up to β₁₀)
            betas = [0.0, 0.0, -11.83e-27, 8.13e-41, -9.5e-56, 2.0e-70, 0.0, 0.0, 0.0, 0.0]
            medium = Medium(0.15, 0.11, betas, 0.0, 835e-9)
            @test length(medium.betas) == 10
        end
    end
    
    @testset "RamanModel" begin
        @testset "Type Hierarchy" begin
            @test BlowWood <: RamanModel
            @test LinAgrawal <: RamanModel
            @test Hollenbeck <: RamanModel
        end
        
        @testset "Instantiation" begin
            bw = BlowWood()
            la = LinAgrawal()
            hc = Hollenbeck()
            
            @test bw isa RamanModel
            @test la isa RamanModel
            @test hc isa RamanModel
        end
    end
    
    @testset "SimParams" begin
        @testset "Default Constructor" begin
            medium = Medium(0.15, 0.11, [0.0, 0.0, -11.83e-27], 0.0, 835e-9)
            
            # All defaults
            params = SimParams(medium=medium)
            @test params.medium === medium
            @test params.N == 2^12
            @test params.n_saves == 200
            @test params.raman == true
            @test params.shock == true
            @test params.raman_model isa Hollenbeck
            @test params.fr == 0.18
            @test params.reltol == 1e-6
            @test params.abstol == 1e-9
        end
        
        @testset "Custom Parameters" begin
            medium = Medium(0.15, 0.11, [0.0], 0.0, 835e-9)
            
            params = SimParams(
                medium=medium,
                N=2^10,
                n_saves=100,
                raman=false,
                shock=false,
                raman_model=BlowWood(),
                fr=0.245,
                reltol=1e-5,
                abstol=1e-8
            )
            
            @test params.N == 2^10
            @test params.n_saves == 100
            @test params.raman == false
            @test params.shock == false
            @test params.raman_model isa BlowWood
            @test params.fr == 0.245
            @test params.reltol == 1e-5
            @test params.abstol == 1e-8
        end
        
        @testset "Validation - n_saves" begin
            medium = Medium(0.15, 0.11, [0.0], 0.0, 835e-9)
            @test_throws ArgumentError SimParams(medium=medium, n_saves=0)
            @test_throws ArgumentError SimParams(medium=medium, n_saves=-10)
        end
        
        @testset "Validation - Raman Fraction" begin
            medium = Medium(0.15, 0.11, [0.0], 0.0, 835e-9)
            @test_throws ArgumentError SimParams(medium=medium, fr=-0.1)
            @test_throws ArgumentError SimParams(medium=medium, fr=1.5)
            
            # Boundary values should work
            params1 = SimParams(medium=medium, fr=0.0)
            params2 = SimParams(medium=medium, fr=1.0)
            @test params1.fr == 0.0
            @test params2.fr == 1.0
        end
        
        @testset "N Power of 2 Warning" begin
            medium = Medium(0.15, 0.11, [0.0], 0.0, 835e-9)
            
            # Power of 2 - no warning
            params = SimParams(medium=medium, N=2^12)
            @test params.N == 2^12
            
            # Non-power of 2 - should warn but work
            params2 = SimParams(medium=medium, N=1000)
            @test params2.N == 1000
        end
        
        @testset "Different Raman Models" begin
            medium = Medium(0.15, 0.11, [0.0], 0.0, 835e-9)
            
            params_bw = SimParams(medium=medium, raman_model=BlowWood())
            params_la = SimParams(medium=medium, raman_model=LinAgrawal())
            params_hc = SimParams(medium=medium, raman_model=Hollenbeck())
            
            @test params_bw.raman_model isa BlowWood
            @test params_la.raman_model isa LinAgrawal
            @test params_hc.raman_model isa Hollenbeck
        end
    end
end
