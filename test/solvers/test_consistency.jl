using Test
using JuGNLSE

@testset "Solvers: Consistency" begin
    # Parameters
    N = 1024
    T = 20e-12
    lambda0 = 1550e-9
    grid = create_grid(N, T, lambda0)

    T0 = 1e-12
    P0 = 100.0
    pulse = gaussian_pulse(grid, T0, P0)

    beta2 = -20e-27
    gamma = 2.0
    L = 0.01
    medium = Medium(L, gamma, [beta2], 0.0, lambda0)

    params = SimParams(; medium=medium, dz=L / 100)

    res_erk = solve(pulse, params; method=:ERK4IP, rtol=1e-10)
    res_rk4 = solve(pulse, params; method=:RK4IP)
    res_ssfm = solve(pulse, params; method=:SSFM)

    # Compare final fields
    At_erk = res_erk.At[:, end]
    At_rk4 = res_rk4.At[:, end]
    At_ssfm = res_ssfm.At[:, end]

    # Compare peak powers
    @test peak_power(At_erk) ≈ peak_power(At_rk4) rtol = 1e-3
    @test peak_power(At_erk) ≈ peak_power(At_ssfm) rtol = 1e-2

    # Compare shapes using correlation
    corr_rk4 =
        abs(sum(conj.(At_erk) .* At_rk4)) /
        (sqrt(sum(abs2.(At_erk))) * sqrt(sum(abs2.(At_rk4))))
    @test corr_rk4 ≈ 1.0 rtol = 0.02

    corr_ssfm =
        abs(sum(conj.(At_erk) .* At_ssfm)) /
        (sqrt(sum(abs2.(At_erk))) * sqrt(sum(abs2.(At_ssfm))))
    @test corr_ssfm ≈ 1.0 rtol = 0.02
end
