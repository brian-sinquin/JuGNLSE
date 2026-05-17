using Test
using JuGNLSE

@testset "JuGNLSE.jl" begin
    include("test_unit.jl")
    include("test_api.jl")
    include("test_solvers.jl")
    include("test_physics.jl")
end
