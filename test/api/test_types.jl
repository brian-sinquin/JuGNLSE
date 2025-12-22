using Test
using JuGNLSE

@testset "API Types" begin
    @testset "Medium" begin
        m = Medium(1.0, 2.0, [-20e-27, 1e-40], 0.1, 1550e-9)
        @test m.length == 1.0
        @test m.gamma == 2.0
        @test m.betas == [-20e-27, 1e-40]
        @test m.alpha == 0.1
        @test m.lambda0 == 1550e-9
    end

    @testset "SimParams" begin
        m = Medium(1.0, 2.0, [-20e-27], 0.0, 1550e-9)
        p = SimParams(; medium=m, n_saves=100, raman=true)
        @test p.medium === m
        @test p.n_saves == 100
        @test p.raman == true
        @test p.shock == true # default is true
    end
end
