using Test
using JuGNLSE

@testset "API Pulses" begin
    grid = create_grid(1024, 10e-12, 1550e-9)

    @testset "Sech Pulse" begin
        duration = 100e-15
        power = 100.0
        p = sech_pulse(grid, duration, power)
        @test peak_power(p.At) ≈ power rtol = 1e-3
        # By default, duration is FWHM
        f = fwhm(abs2.(p.At), grid.t)
        @test f ≈ duration rtol = 1e-2
    end

    @testset "Gaussian Pulse" begin
        duration = 100e-15
        power = 100.0
        p = gaussian_pulse(grid, duration, power)
        @test peak_power(p.At) ≈ power rtol = 1e-3
        # By default, duration is FWHM
        f = fwhm(abs2.(p.At), grid.t)
        @test f ≈ duration rtol = 1e-2
    end
end
