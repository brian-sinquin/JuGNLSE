using Test
using JuGNLSE

@testset "Grid Utilities" begin
    N = 1024
    T = 10e-12
    lambda0 = 835e-9
    grid = create_grid(N, T, lambda0)

    @test grid.N == N
    @test grid.N * grid.dt ≈ T
    @test grid.dt ≈ T / N
    @test length(grid.t) == N
    @test length(grid.omega) == N

    # Check omega spacing
    dw = 2π / T
    @test grid.omega[2] - grid.omega[1] ≈ dw rtol = 1e-10

    # Check center frequency
    c = JuGNLSE.SPEED_OF_LIGHT
    w0 = 2π * c / lambda0
    @test grid.omega0 ≈ w0 rtol = 1e-10

    # Check that omega is in FFT order (0 at index 1)
    @test grid.omega[1] ≈ 0.0 atol = 1e-10
end
