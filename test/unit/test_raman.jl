# Unit tests for Raman response models
using Test
using JuGNLSE
using FFTW

@testset "Raman Response" begin
    grid = create_grid(2^12, 10e-12, 835e-9)
    
    @testset "Blow-Wood Model" begin
        h_R, fr = raman_response(grid, BlowWood())
        
        @test length(h_R) == grid.N
        @test fr == 0.18
        
        # Response should be causal (zero for t < 0)
        negative_time_indices = grid.t .< 0
        @test all(h_R[negative_time_indices] .== 0)
        
        # Response should be normalized
        integral = sum(h_R) * grid.dt
        @test integral ≈ 1.0 rtol=1e-2
        
        # Peak should be around τ1 ≈ 12.2 fs
        peak_idx = argmax(h_R)
        peak_time = grid.t[peak_idx]
        @test 0 < peak_time < 50e-15  # Should peak within ~50 fs
    end
    
    @testset "Lin-Agrawal Model" begin
        h_R, fr = raman_response(grid, LinAgrawal())
        
        @test length(h_R) == grid.N
        @test fr == 0.245
        
        # Causal
        negative_time_indices = grid.t .< 0
        @test all(h_R[negative_time_indices] .== 0)
        
        # Normalized
        integral = sum(h_R) * grid.dt
        @test integral ≈ 1.0 rtol=1e-2
    end
    
    @testset "Hollenbeck Model" begin
        h_R, fr = raman_response(grid, Hollenbeck())
        
        @test length(h_R) == grid.N
        @test fr ≈ 0.20 rtol=0.01
        
        # Causal
        negative_time_indices = grid.t .< 0
        @test all(h_R[negative_time_indices] .== 0)
        
        # Normalized
        integral = sum(h_R) * grid.dt
        @test integral ≈ 1.0 rtol=1e-2
    end
    
    @testset "Frequency Domain Transformation" begin
        h_R, fr = raman_response(grid, BlowWood())
        RW = raman_response_frequency(h_R, grid)
        
        @test length(RW) == grid.N
        @test all(isfinite.(RW))
        
        # Basic check: frequency response should be non-zero
        @test sum(abs2.(RW)) > 0
    end
    
    @testset "Model Comparison" begin
        # Different models should give different responses
        h_bw, fr_bw = raman_response(grid, BlowWood())
        h_la, fr_la = raman_response(grid, LinAgrawal())
        h_hc, fr_hc = raman_response(grid, Hollenbeck())
        
        # Raman fractions differ
        @test fr_bw != fr_la
        
        # Response shapes differ
        @test !(h_bw ≈ h_la)
    end
    
    @testset "Response Properties" begin
        h_R, fr = raman_response(grid, BlowWood())
        
        # Response should be causal and have a peak
        @test maximum(h_R) > 0
        
        # Response should decay to zero
        late_time_idx = findfirst(grid.t .> 1e-12)  # After 1 ps
        if late_time_idx !== nothing
            @test h_R[late_time_idx] < 0.01 * maximum(h_R)
        end
    end
end
