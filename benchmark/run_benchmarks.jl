"""
Comprehensive benchmark suite for JuGNLSE.jl

This script runs detailed benchmarks and generates a performance report.
Tests all three solvers: SSFM, RK4IP, and ERK4IP with adaptive stepping.

Run with: julia --project=. benchmark/run_benchmarks.jl
"""

using JuGNLSE
using BenchmarkTools
using Printf
using Dates
using FFTW

println("="^70)
println("JuGNLSE.jl Performance Benchmark Suite")
println("Julia Version: $(VERSION)")
println("Date: $(Dates.now())")
println("="^70)

# Results storage
results = Dict{String, Any}()

# Helper function to format time
function format_time(t_ns)
    if t_ns < 1e3
        return @sprintf("%.2f ns", t_ns)
    elseif t_ns < 1e6
        return @sprintf("%.2f μs", t_ns / 1e3)
    elseif t_ns < 1e9
        return @sprintf("%.2f ms", t_ns / 1e6)
    else
        return @sprintf("%.2f s", t_ns / 1e9)
    end
end

# Helper function to format memory
function format_memory(bytes)
    if bytes < 1024
        return @sprintf("%.0f B", bytes)
    elseif bytes < 1024^2
        return @sprintf("%.2f KB", bytes / 1024)
    elseif bytes < 1024^3
        return @sprintf("%.2f MB", bytes / 1024^2)
    else
        return @sprintf("%.2f GB", bytes / 1024^3)
    end
end

# 1. Component Benchmarks
println("\n" * "="^70)
println("1. COMPONENT BENCHMARKS")
println("="^70)

# Grid creation
println("\n▶ Grid Creation")
for N in [2^10, 2^12, 2^14]
    b = @benchmark create_grid($N, 10e-12, 835e-9)
    println("  N=$N: $(format_time(median(b.times))) | $(format_memory(b.memory))")
    results["grid_$N"] = median(b.times)
end

# Pulse generation
println("\n▶ Pulse Generation")
grid = create_grid(2^12, 10e-12, 835e-9)
b_sech = @benchmark sech_pulse($grid, 50e-15, 10000.0, 835e-9)
b_gauss = @benchmark gaussian_pulse($grid, 50e-15, 10000.0, 835e-9)
println("  Sech:     $(format_time(median(b_sech.times))) | $(format_memory(b_sech.memory))")
println("  Gaussian: $(format_time(median(b_gauss.times))) | $(format_memory(b_gauss.memory))")
results["pulse_sech"] = median(b_sech.times)
results["pulse_gauss"] = median(b_gauss.times)

# FFT operations
println("\n▶ FFT Operations (N=2^12)")
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
b_fft = @benchmark fft($(pulse.At))
b_ifft = @benchmark ifft($(pulse.Aw))
println("  FFT:  $(format_time(median(b_fft.times))) | $(format_memory(b_fft.memory))")
println("  IFFT: $(format_time(median(b_ifft.times))) | $(format_memory(b_ifft.memory))")
results["fft"] = median(b_fft.times)
results["ifft"] = median(b_ifft.times)

# Operators
println("\n▶ Operator Construction")
medium = Medium(0.15, 0.11, [0.0, 0.0, -11.83e-27, 8.13e-41], 0.0, 835e-9)
b_disp = @benchmark dispersion_operator($grid, $medium)
println("  Dispersion: $(format_time(median(b_disp.times))) | $(format_memory(b_disp.memory))")

b_raman_bw = @benchmark raman_response($grid, BlowWood())
b_raman_la = @benchmark raman_response($grid, LinAgrawal())
b_raman_hc = @benchmark raman_response($grid, Hollenbeck())
println("  Raman (Blow-Wood):  $(format_time(median(b_raman_bw.times)))")
println("  Raman (Lin-Agrawal): $(format_time(median(b_raman_la.times)))")
println("  Raman (Hollenbeck):  $(format_time(median(b_raman_hc.times)))")
results["raman_bw"] = median(b_raman_bw.times)

# 2. Method Comparison
println("\n" * "="^70)
println("2. METHOD COMPARISON (SSFM vs RK4IP vs ERK4IP)")
println("="^70)

for (N, n_saves) in [(2^10, 20), (2^11, 50)]
    println("\n▶ N=$N, n_saves=$n_saves")
    
    medium_test = Medium(0.1, 0.11, [0.0, 0.0, -11.83e-27], 0.0, 835e-9)
    grid_test = create_grid(N, 10e-12, 835e-9)
    pulse_test = sech_pulse(grid_test, 50e-15, 5000.0, 835e-9)
    
    params_test = SimParams(medium=medium_test, N=N, n_saves=n_saves, raman=false, shock=false)
    
    # Warmup
    solve(pulse_test, params_test, method=:ssfm, progress=false)
    solve(pulse_test, params_test, method=:rk4ip, progress=false)
    solve(pulse_test, params_test, method=:erk4ip, progress=false)
    
    b_ssfm = @benchmark solve($pulse_test, $params_test, method=:ssfm, progress=false) samples=5
    b_rk4ip = @benchmark solve($pulse_test, $params_test, method=:rk4ip, progress=false) samples=5
    b_erk4ip = @benchmark solve($pulse_test, $params_test, method=:erk4ip, progress=false) samples=5
    
    println("  SSFM:   $(format_time(median(b_ssfm.times))) | $(format_memory(b_ssfm.memory))")
    println("  RK4IP:  $(format_time(median(b_rk4ip.times))) | $(format_memory(b_rk4ip.memory))")
    println("  ERK4IP: $(format_time(median(b_erk4ip.times))) | $(format_memory(b_erk4ip.memory))")
    println("  RK4IP/SSFM ratio: $(round(median(b_rk4ip.times) / median(b_ssfm.times), digits=2))x")
    println("  ERK4IP/SSFM ratio: $(round(median(b_erk4ip.times) / median(b_ssfm.times), digits=2))x")
    
    results["ssfm_$(N)"] = median(b_ssfm.times)
    results["rk4ip_$(N)"] = median(b_rk4ip.times)
    results["erk4ip_$(N)"] = median(b_erk4ip.times)
end

# 3. Scaling Analysis
println("\n" * "="^70)
println("3. SCALING ANALYSIS")
println("="^70)

println("\n▶ Performance vs Grid Size (SSFM)")
println("  N      | Time      | Memory    | Time/Point")
println("  " * "-"^50)

scaling_data = []
for N in [2^8, 2^10, 2^12]
    medium_scale = Medium(0.05, 0.11, [0.0, 0.0, -11.83e-27], 0.0, 835e-9)
    grid_scale = create_grid(N, 10e-12, 835e-9)
    pulse_scale = sech_pulse(grid_scale, 50e-15, 5000.0, 835e-9)
    params_scale = SimParams(medium=medium_scale, N=N, n_saves=10, raman=false, shock=false)
    
    b = @benchmark solve($pulse_scale, $params_scale, method=:ssfm, progress=false) samples=3
    
    time_per_point = median(b.times) / N
    push!(scaling_data, (N, median(b.times), b.memory, time_per_point))
    
    @printf("  %-6d | %-9s | %-9s | %.2f ns\n", 
            N, format_time(median(b.times)), format_memory(b.memory), time_per_point)
end

# 4. Physics Effects Benchmark
println("\n" * "="^70)
println("4. PHYSICS EFFECTS BENCHMARK (N=2^11)")
println("="^70)

medium_phys = Medium(0.15, 0.11, [0.0, 0.0, -11.83e-27, 8.13e-41], 0.0, 835e-9)
grid_phys = create_grid(2^11, 10e-12, 835e-9)
pulse_phys = sech_pulse(grid_phys, 50e-15, 10000.0, 835e-9)

configs = [
    ("Linear only", false, false),
    ("Kerr only", false, false),
    ("Kerr + Raman", true, false),
    ("Full physics", true, true)
]

println("\n  Configuration      | Time      | Memory")
println("  " * "-"^50)

for (name, raman, shock) in configs
    params = SimParams(medium=medium_phys, N=2^11, n_saves=30, 
                      raman=raman, shock=shock, raman_model=Hollenbeck())
    
    # Warmup
    solve(pulse_phys, params, method=:rk4ip, progress=false)
    
    b = @benchmark solve($pulse_phys, $params, method=:rk4ip, progress=false) samples=3 seconds=20
    
    @printf("  %-18s | %-9s | %s\n", 
            name, format_time(median(b.times)), format_memory(b.memory))
    
    results["physics_$name"] = median(b.times)
end

# 5. Summary and Report
println("\n" * "="^70)
println("5. PERFORMANCE SUMMARY")
println("="^70)

println("\n▶ Key Metrics:")
println("  • Grid creation (2^12):     $(format_time(results["grid_2048"]))")
println("  • Pulse generation:         $(format_time(results["pulse_sech"]))")
println("  • FFT (2^12):               $(format_time(results["fft"]))")
println("  • Raman response:           $(format_time(results["raman_bw"]))")
println("  • SSFM solve (2^11, 50 pts): $(format_time(results["ssfm_2048"]))")
println("  • RK4IP solve (2^11, 50 pts): $(format_time(results["rk4ip_2048"]))")

println("\n▶ Performance Targets:")
target_ssfm = results["ssfm_2048"] < 500e6  # < 500 ms
target_rk4ip = results["rk4ip_2048"] < 3e9  # < 3 s
target_fft = results["fft"] < 1e6           # < 1 ms

println("  • SSFM < 500 ms:    $(target_ssfm ? "✓ PASS" : "✗ FAIL")")
println("  • RK4IP < 3 s:      $(target_rk4ip ? "✓ PASS" : "✗ FAIL")")
println("  • FFT < 1 ms:       $(target_fft ? "✓ PASS" : "✗ FAIL")")

# Save results to file
println("\n" * "="^70)
println("Saving benchmark results...")

open("benchmark_results.txt", "w") do io
    println(io, "JuGNLSE.jl Benchmark Results")
    println(io, "Date: $(Dates.now())")
    println(io, "Julia Version: $(VERSION)")
    println(io, "="^70)
    println(io)
    
    for (key, value) in sort(collect(results))
        println(io, "$key: $(format_time(value))")
    end
end

println("✓ Results saved to: benchmark_results.txt")

println("\n" * "="^70)
println("Benchmark Suite Complete!")
println("="^70)
