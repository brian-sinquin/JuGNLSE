using Test
using JuGNLSE

@testset "JuGNLSE.jl" begin
    @testset "Unit Tests" begin
        include("unit/test_physics.jl")
        include("unit/test_analysis.jl")
        include("unit/test_grid.jl")
    end

    @testset "API Tests" begin
        include("api/test_types.jl")
        include("api/test_pulses.jl")
    end

    @testset "Physics Tests" begin
        include("physics/test_dispersion.jl")
        include("physics/test_nonlinearity.jl")
        include("physics/test_solitons.jl")
        include("physics/test_shock.jl")
        include("physics/test_raman.jl")
    end

    @testset "Solver Consistency" begin
        include("solvers/test_consistency.jl")
    end

    # include("comparison_tests/dudley.jl")
end
