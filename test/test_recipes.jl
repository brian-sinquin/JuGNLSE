using Test
using Soliton
using Plots

@testset "Plot Recipes" begin
    grid = create_grid(2^9, 5e-12, 1550e-9)
    pulse = sech_pulse(grid, 100.0, 100e-15)
    medium = Medium(0.05, 0.0011, 0.0, [-21.5e-27], 1550e-9)
    sol = solve(
        pulse, SimParams(; medium=medium, z_saves=5, raman_model=nothing); progress=false
    )

    p = plot(sol)
    @test p isa Plots.Plot
    @test length(p.subplots) == 4
end

@testset "Recipe spectral wavelength alignment" begin
    grid = create_grid(128, 2e-12, 1550e-9)
    dw = 2π / (grid.N * grid.dt)
    tone = cis.(-7dw .* grid.t)
    pulse = Pulse(tone, ifft(tone), grid)
    medium = Medium(length=0.01, gamma=0.0, betas=[0.0], lambda0=grid.lambda0)
    sol = solve(pulse, SimParams(; medium, z_saves=2, raman_model=nothing,
        self_steepening=false); progress=false)
    p = plot(sol)
    spectral = p.series_list[end]
    @test spectral[:x][argmax(spectral[:y])] ≈ 2π * c / (grid.omega0 + 7dw) * 1e9
end
