using Test
using JuGNLSE
using FFTW

@testset "Analysis Utilities" begin
    @testset "Pulse Metrics" begin
        N = 1024
        dt = 10e-15
        t = collect(range(-N / 2, N / 2 - 1; length=N)) .* dt

        # Gaussian pulse
        T0 = 100e-15
        P0 = 1.0
        At = sqrt(P0) .* exp.(-t .^ 2 ./ (2 * T0^2))

        # Peak power
        @test peak_power(At) ≈ P0 rtol = 1e-10

        # Energy: E = P0 * sqrt(pi) * T0 * sqrt(2) ? No, for exp(-t^2/2T0^2), E = P0 * sqrt(pi) * T0 * sqrt(2) is wrong.
        # Integral of exp(-t^2/T0^2) is sqrt(pi) * T0.
        E_analytical = P0 * sqrt(π) * T0
        @test pulse_energy(At, dt) ≈ E_analytical rtol = 1e-5

        # FWHM
        # For Gaussian exp(-t^2/2T0^2), P(t) = exp(-t^2/T0^2)
        # FWHM = 2*sqrt(ln2)*T0
        fwhm_val = fwhm(abs2.(At), t)
        @test fwhm_val ≈ 2 * sqrt(log(2)) * T0 rtol = 1e-3
    end

    @testset "Spectral Metrics" begin
        N = 1024
        dt = 10e-15
        t = collect(range(-N / 2, N / 2 - 1; length=N)) .* dt
        T0 = 100e-15
        At = exp.(-t .^ 2 ./ (2 * T0^2))

        # FFT
        Aw = fftshift(fft(ifftshift(At)))
        dw = 2π / (N * dt)
        omega = collect(range(-N / 2, N / 2 - 1; length=N)) .* dw

        # Spectral bandwidth
        # Aw is proportional to exp(-omega^2 * T0^2 / 2)
        # |Aw|^2 is proportional to exp(-omega^2 * T0^2)
        # FWHM_omega = 2 * sqrt(ln2) / T0
        bw = spectral_bandwidth(Aw, omega)
        @test bw ≈ 2 * sqrt(log(2)) / T0 rtol = 1e-3

        # TBP
        tbp = time_bandwidth_product(At, Aw, t, omega)
        # For Gaussian, TBP = (Δt_fwhm * Δω_fwhm) / (4π)
        # = (2*sqrt(ln2)*T0 * 2*sqrt(ln2)/T0) / (4π) = 4*ln2 / 4π = ln2/π ≈ 0.2206
        @test tbp ≈ log(2) / π rtol = 1e-3
    end
end
