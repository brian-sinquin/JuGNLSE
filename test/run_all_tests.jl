"""
Run all tests including standard tests, plotting tests, and benchmarks.

Usage:
    julia test/run_all_tests.jl [options]

Options:
    --quick      Skip benchmarks (faster testing)
    --plots      Run plotting tests only
    --bench      Run benchmarks only
    --all        Run everything (default)
"""

using Pkg
Pkg.activate(".")

println("="^60)
println("JuGNLSE.jl Test Suite")
println("="^60)

# Parse command line arguments
args = ARGS
run_standard = true
run_plots = true
run_bench = true

if "--quick" in args
    run_bench = false
    println("Quick mode: Skipping benchmarks")
elseif "--plots" in args
    run_standard = false
    run_bench = false
    println("Running plotting tests only")
elseif "--bench" in args
    run_standard = false
    run_plots = false
    println("Running benchmarks only")
end

# Track timing
start_time = time()

# Run standard tests
if run_standard
    println("\n" * "="^60)
    println("Running Standard Tests")
    println("="^60)
    include("runtests.jl")
end

# Run plotting tests
if run_plots
    println("\n" * "="^60)
    println("Running Plotting Tests")
    println("="^60)
    include("test_plotting.jl")
end

# Run benchmarks
if run_bench
    println("\n" * "="^60)
    println("Running Performance Benchmarks")
    println("="^60)
    include("test_benchmark.jl")
end

# Summary
elapsed = time() - start_time
println("\n" * "="^60)
println("Test Suite Complete!")
println("="^60)
println("Total time: $(round(elapsed, digits=2)) seconds")
println("\nAll tests passed! ✓")
println("="^60)
