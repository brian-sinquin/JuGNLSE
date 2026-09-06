using Test, Soliton, FFTW, LinearAlgebra

@testset "Audit Critical Fixes" begin

    @testset "C2 solution pulse spectral ordering" begin
        grid = create_grid(256, 10e-12, 1550e-9)
        pulse = gaussian_pulse(grid, 1.0, 1e-12)
        medium = Medium(length=0.1, gamma=0.0, betas=[0.0], lambda0=1550e-9)
        sol = solve(pulse, SimParams(medium=medium, z_saves=2,
            raman_model=nothing, self_steepening=false); progress=false)
        result = Pulse(sol)
        # A consistent Pulse must satisfy AW == ifft(At)
        @test norm(result.AW - ifft(result.At)) / norm(result.AW) < 1e-10
    end

    @testset "C3 filter spectral ordering" begin
        grid = create_grid(256, 10e-12, 1550e-9)
        pulse = gaussian_pulse(grid, 1.0, 1e-12)
        vpulse = VectorialPulse(pulse.At, 0.5im .* pulse.At, grid)
        # Broadband passband centered on carrier — should retain >99% energy
        filt = Filter(w -> abs(w - grid.omega0) < 1e13 ? 1.0 : 0.0)
        for input in (pulse, vpulse)
            @test pulse_energy(apply(input, filt)) / pulse_energy(input) > 0.99
        end
    end

    @testset "C4 zero-gain amplifier Kerr normalization" begin
        grid = create_grid(256, 10e-12, 1550e-9)
        pulse = gaussian_pulse(grid, 1.0, 1e-12)
        # With g0=0, amplifying medium Kerr should match passive medium exactly
        passive = Medium(length=0.1, gamma=0.7, betas=[0.0], lambda0=1550e-9)
        active = AmplifyingMedium(length=0.1, gamma=0.7, g0=0.0,
            Esat=1e-9, betas=[0.0], lambda0=1550e-9)
        models = map((passive, active)) do medium
            build_physics_model(grid, SimParams(medium=medium, z_saves=2,
                raman_model=nothing, self_steepening=false))
        end
        rhs = map(m -> copy(m.nonlinear_function(pulse.At, m, 0.03)), models)
        @test norm(rhs[2] - rhs[1]) / norm(rhs[1]) < 1e-12
    end

    @testset "H4 grid validation" begin
        # N < 2 must be rejected
        @test_throws ArgumentError create_grid(0, 10e-12, 1550e-9)
        @test_throws ArgumentError create_grid(1, 10e-12, 1550e-9)
        # Odd N must be rejected
        @test_throws ArgumentError create_grid(3, 10e-12, 1550e-9)
        @test_throws ArgumentError create_grid(5, 10e-12, 1550e-9)
        @test_throws ArgumentError create_grid(127, 10e-12, 1550e-9)
        # Even N >= 2 should work
        grid2 = create_grid(2, 10e-12, 1550e-9)
        @test length(grid2.t) == 2
        @test length(grid2.V) == 2
        grid6 = create_grid(6, 10e-12, 1550e-9)
        @test length(grid6.t) == 6
        @test length(grid6.V) == 6
    end

end
