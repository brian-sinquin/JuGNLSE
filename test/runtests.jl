using JuGNLSE
using Test

@testset "JuGNLSE.jl" begin
    # Unit tests - Test each component individually
    @testset "Unit Tests" begin
        include("unit/test_types.jl")
        include("unit/test_grid.jl")
        include("unit/test_pulse.jl")
        include("unit/test_dispersion.jl")
        include("unit/test_nonlinearity.jl")
        include("unit/test_raman.jl")
    end
    
    # Integration tests - Test solvers and workflows
    @testset "Integration Tests" begin
        include("integration/test_ssfm.jl")
        include("integration/test_rk4ip.jl")
        include("integration/test_erk4ip.jl")
        include("integration/test_energy_conservation.jl")
        include("integration/test_raman_solvers.jl")
    end
    
    # Regression tests - Reproduce published results
    @testset "Regression Tests" begin
        include("regression/test_dudley2006.jl")
        include("regression/test_solitons.jl")
    end
end
