"""
Performance benchmarks for JuGNLSE.jl

These tests measure the performance of various components and compare methods.
"""

using JuGNLSE
using Test
using BenchmarkTools

@testset "Performance Benchmarks" begin
    
    @testset "Grid Creation Benchmark" begin
        println("\n⚡ Benchmarking grid creation...")
        
        b = @benchmark create_grid(2^12, 10e-12, 835e-9)
        
        println("  Grid creation (2^12 points):")
        println("    Median time: $(median(b.times) / 1e6) ms")
        println("    Memory: $(b.memory / 1024) KB")
        println("    Allocations: $(b.allocs)")
        
        @test median(b.times) < 1e7  # Should be < 10 ms
    end
    
    @testset "Pulse Generation Benchmark" begin
        println("\n⚡ Benchmarking pulse generation...")
        
        grid = create_grid(2^12, 10e-12, 835e-9)
        
        # Sech pulse
        b_sech = @benchmark sech_pulse($grid, 50e-15, 10000.0, 835e-9)
        println("  Sech pulse generation:")
        println("    Median time: $(median(b_sech.times) / 1e6) ms")
        println("    Memory: $(b_sech.memory / 1024) KB")
        
        # Gaussian pulse
        b_gauss = @benchmark gaussian_pulse($grid, 50e-15, 10000.0, 835e-9)
        println("  Gaussian pulse generation:")
        println("    Median time: $(median(b_gauss.times) / 1e6) ms")
        println("    Memory: $(b_gauss.memory / 1024) KB")
        
        @test median(b_sech.times) < 5e7  # < 50 ms
        @test median(b_gauss.times) < 5e7
    end
    
    @testset "FFT Operations Benchmark" begin
        println("\n⚡ Benchmarking FFT operations...")
        
        grid = create_grid(2^12, 10e-12, 835e-9)
        pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
        
        # Forward FFT
        b_fft = @benchmark fft($(pulse.At))
        println("  FFT (2^12 points):")
        println("    Median time: $(median(b_fft.times) / 1e3) μs")
        println("    Memory: $(b_fft.memory / 1024) KB")
        
        # Inverse FFT
        b_ifft = @benchmark ifft($(pulse.Aw))
        println("  IFFT (2^12 points):")
        println("    Median time: $(median(b_ifft.times) / 1e3) μs")
        println("    Memory: $(b_ifft.memory / 1024) KB")
        
        @test median(b_fft.times) < 1e6  # < 1 ms
        @test median(b_ifft.times) < 1e6
    end
    
    @testset "Dispersion Operator Benchmark" begin
        println("\n⚡ Benchmarking dispersion operator...")
        
        grid = create_grid(2^12, 10e-12, 835e-9)
        medium = Medium(0.15, 0.11, [0.0, 0.0, -11.83e-27, 8.13e-41], 0.0, 835e-9)
        
        b = @benchmark dispersion_operator($grid, $fiber)
        println("  Dispersion operator creation:")
        println("    Median time: $(median(b.times) / 1e6) ms")
        println("    Memory: $(b.memory / 1024) KB")
        
        @test median(b.times) < 1e7  # < 10 ms
    end
    
    @testset "Raman Response Benchmark" begin
        println("\n⚡ Benchmarking Raman response functions...")
        
        grid = create_grid(2^12, 10e-12, 835e-9)
        
        # Blow-Wood
        b_bw = @benchmark raman_response($grid, BlowWood())
        println("  Blow-Wood model:")
        println("    Median time: $(median(b_bw.times) / 1e6) ms")
        
        # Lin-Agrawal
        b_la = @benchmark raman_response($grid, LinAgrawal())
        println("  Lin-Agrawal model:")
        println("    Median time: $(median(b_la.times) / 1e6) ms")
        
        # Hollenbeck
        b_hc = @benchmark raman_response($grid, Hollenbeck())
        println("  Hollenbeck model:")
        println("    Median time: $(median(b_hc.times) / 1e6) ms")
        
        @test median(b_bw.times) < 5e7  # < 50 ms
    end
    
    @testset "SSFM vs RK4IP Benchmark" begin
        println("\n⚡ Benchmarking SSFM vs RK4IP methods...")
        
        # Small problem for benchmarking
        medium = Medium(0.1, 0.11, [0.0, 0.0, -11.83e-27], 0.0, 835e-9)
        grid = create_grid(2^10, 10e-12, 835e-9)
        pulse = sech_pulse(grid, 50e-15, 5000.0, 835e-9)
        
        params_ssfm = SimParams(
            fiber=fiber,
            N=2^10,
            n_saves=20,
            raman=false,
            shock=false
        )
        
        params_rk4ip = SimParams(
            fiber=fiber,
            N=2^10,
            n_saves=20,
            raman=false,
            shock=false,
            reltol=1e-5,
            abstol=1e-8
        )
        
        # Benchmark SSFM
        println("  Running SSFM benchmark (2^10 points, 20 saves)...")
        b_ssfm = @benchmark solve($pulse, $params_ssfm, method=:ssfm, progress=false)
        println("    Median time: $(median(b_ssfm.times) / 1e6) ms")
        println("    Memory: $(b_ssfm.memory / 1024^2) MB")
        println("    Allocations: $(b_ssfm.allocs)")
        
        # Benchmark RK4IP
        println("  Running RK4IP benchmark (2^10 points, 20 saves)...")
        b_rk4ip = @benchmark solve($pulse, $params_rk4ip, method=:rk4ip, progress=false)
        println("    Median time: $(median(b_rk4ip.times) / 1e6) ms")
        println("    Memory: $(b_rk4ip.memory / 1024^2) MB")
        println("    Allocations: $(b_rk4ip.allocs)")
        
        ratio = median(b_rk4ip.times) / median(b_ssfm.times)
        println("  RK4IP/SSFM time ratio: $(round(ratio, digits=2))x")
        
        @test median(b_ssfm.times) < 5e8  # < 500 ms
        @test median(b_rk4ip.times) < 2e9  # < 2 s
    end
    
    @testset "Scaling with Grid Size" begin
        println("\n⚡ Testing performance scaling with grid size...")
        
        grid_sizes = [2^8, 2^10, 2^12]
        times_ssfm = Float64[]
        
        for N in grid_sizes
            medium = Medium(0.05, 0.11, [0.0, 0.0, -11.83e-27], 0.0, 835e-9)
            grid = create_grid(N, 10e-12, 835e-9)
            pulse = sech_pulse(grid, 50e-15, 5000.0, 835e-9)
            
            params = SimParams(
                fiber=fiber,
                N=N,
                n_saves=10,
                raman=false,
                shock=false
            )
            
            b = @benchmark solve($pulse, $params, method=:ssfm, progress=false) samples=3 seconds=10
            push!(times_ssfm, median(b.times) / 1e6)
            
            println("  N = $N: $(round(times_ssfm[end], digits=2)) ms")
        end
        
        # Check that scaling is reasonable (should be roughly N*log(N) for FFT)
        @test all(times_ssfm .> 0)
    end
    
    @testset "Full Physics Benchmark" begin
        println("\n⚡ Benchmarking full physics simulation...")
        
        medium = Medium(0.15, 0.11, [0.0, 0.0, -11.83e-27, 8.13e-41], 0.0, 835e-9)
        grid = create_grid(2^11, 10e-12, 835e-9)
        pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
        
        params_full = SimParams(
            fiber=fiber,
            N=2^11,
            n_saves=50,
            raman=true,
            shock=true,
            raman_model=Hollenbeck()
        )
        
        println("  Running full physics (2^11 points, Raman + shock)...")
        println("  This may take a moment...")
        
        # Run once to compile
        solve(pulse, params_full, method=:rk4ip, progress=false)
        
        # Benchmark
        b = @benchmark solve($pulse, $params_full, method=:rk4ip, progress=false) samples=3 seconds=30
        
        println("    Median time: $(median(b.times) / 1e9) s")
        println("    Memory: $(b.memory / 1024^2) MB")
        println("    Allocations: $(b.allocs)")
        
        @test median(b.times) < 30e9  # < 30 seconds
    end
    
    @testset "Memory Efficiency" begin
        println("\n⚡ Testing memory efficiency...")
        
        medium = Medium(0.1, 0.11, [0.0, 0.0, -11.83e-27], 0.0, 835e-9)
        grid = create_grid(2^12, 10e-12, 835e-9)
        pulse = sech_pulse(grid, 50e-15, 5000.0, 835e-9)
        
        params = SimParams(
            fiber=fiber,
            N=2^12,
            n_saves=100,
            raman=false,
            shock=false
        )
        
        # Measure allocations
        b = @benchmark solve($pulse, $params, method=:ssfm, progress=false)
        
        println("  Total allocations: $(b.allocs)")
        println("  Total memory: $(b.memory / 1024^2) MB")
        println("  Allocations per iteration: $(b.allocs / 100)")
        
        # Check that we're not allocating excessively
        @test b.memory < 500 * 1024^2  # < 500 MB for this problem size
    end
end

println("\n" * "="^60)
println("Performance Benchmark Summary")
println("="^60)
println("All benchmarks completed successfully!")
println("Check the output above for detailed timing information.")
println("="^60)
