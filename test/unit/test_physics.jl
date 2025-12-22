using Test
using JuGNLSE

@testset "Physics Utilities" begin
    @testset "Unit Conversions" begin
        # Loss conversions
        # 0.2 dB/km is typical for SMF-28
        alpha_dbkm = 0.2
        alpha_npm = convert_loss(alpha_dbkm, (:dB, :km), (:linear, :m))
        # 0.2 dB/km = 0.2 / (10 * log10(exp(1))) / 1000 Np/m ≈ 4.605e-5 Np/m
        @test alpha_npm ≈ alpha_dbkm / (10 * log10(exp(1))) / 1000 rtol = 1e-10

        # Round trip
        @test convert_loss(alpha_npm, (:linear, :m), (:dB, :km)) ≈ alpha_dbkm rtol = 1e-10

        # Wavelength/Frequency
        c = JuGNLSE.SPEED_OF_LIGHT
        @test wavelength_to_frequency(1550e-9) ≈ c / 1550e-9 rtol = 1e-10
        @test frequency_to_wavelength(c / 1550e-9) ≈ 1550e-9 rtol = 1e-10

        # dB/Linear
        @test db_to_linear(10.0) ≈ 10.0 rtol = 1e-10
        @test linear_to_db(10.0) ≈ 10.0 rtol = 1e-10
        @test db_to_linear(-3.0103) ≈ 0.5 rtol = 1e-4
    end

    @testset "Soliton Math" begin
        beta2 = -20e-27
        gamma = 2.0
        T0 = 100e-15

        P0 = calculate_soliton_power(beta2, gamma, T0)
        # P0 = |beta2| / (gamma * T0^2) = 20e-27 / (2.0 * 100e-15^2) = 20e-27 / (2.0 * 1e-26) = 1.0 W
        @test P0 ≈ 1.0 rtol = 1e-10

        @test soliton_order(1.0, beta2, gamma, T0) ≈ 1.0 rtol = 1e-10
        @test soliton_order(4.0, beta2, gamma, T0) ≈ 2.0 rtol = 1e-10
    end
end
