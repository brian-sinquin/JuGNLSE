using Test
using CSV
using DataFrames
using JuGNLSE

@testset "Supercontinuum Generation - Dudley Test" begin
    # === Simulation Parameters ===
    n = 2^13                   # Number of grid points
    twidth = 12.5e-12          # Width of time window [s]
    wavelength = 835e-9        # Reference wavelength [m]

    # === Input Pulse ===
    power = 10000              # Peak power of input [W]
    t0 = 28.4e-15              # Duration of input [s]

    # === Fiber Parameters ===
    flength = 0.15             # Fiber length [m]
    # JuGNLSE uses betas[1]=β₂, betas[2]=β₃, etc. (skips β₀, β₁)
    betas = [
        -1.1830e-026,
        8.1038e-041,
        -9.5205e-056,
        2.0737e-070,
        -5.3943e-085,
        1.3486e-099,
        -2.5495e-114,
        3.0524e-129,
        -1.7140e-144,
    ]     # Dispersion coefficients β₂, β₃, ... [s^n/m]
    gamma = 0.11               # Nonlinear coefficient [1/W/m]
    loss = 0.0                 # Loss [dB/m]

    # Create JuGNLSE grid
    grid = create_grid(n, twidth, wavelength)

    # Create pulse using JuGNLSE
    pulse = sech_pulse(grid, t0, power; T0=true)

    # Define the medium
    medium = Medium(flength, gamma, betas, loss, wavelength)

    # === Problem Setup ===
    nsaves = 200               # Number of length steps to save field at

    # Create simulation parameters with Raman and self-steepening
    params = SimParams(;
        medium=medium,
        n_saves=nsaves,
        raman=true,
        shock=true,
        raman_model=BlowWood(0.18),
        fr=0.18,
    )

    # Solve the GNLSE
    sol = solve(pulse, params; rtol=1e-6)

    # === Load Reference Data ===
    fn = joinpath(dirname(@__FILE__), "data/table_dudley_test_t.csv")
    dat = CSV.read(fn, DataFrame)

    t_dudley = dat.t
    It_dudley = parse.(ComplexF64, dat.At) .|> abs2

    # === Compare Results ===
    # JuGNLSE stores results as (points × saves), last column is final output
    I = reverse(abs2.(sol.At[:, end]))  # Reverse the simulated intensity
    err = 1 / length(I) * sum(abs.(I .- It_dudley) / maximum(I))  # Compute error

    # Test if the error is within acceptable bounds (1.5% tolerance for numerical variations)
    @test err < 1.5 / 100
end
