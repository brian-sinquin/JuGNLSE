
using Test
using LinearAlgebra
using FFTW

# Mock JuGNLSE types and functions for testing PropagationSteps
# Only define the fields strictly necessary for the tests

module MockJuGNLSE
    using ..Test
    using ..LinearAlgebra
    using ..FFTW

    export Grid, Pulse, Medium, SimParams, GNLSEProblem, ConstantGamma, Solution, TaylorDispersion
    export create_grid, sech_pulse, solve, pulse_energy

    struct Grid
        N::Int
        dt::Float64
        t::Vector{Float64}
        W::Vector{Float64}
        omega0::Float64
    end

    function create_grid(N, time_window, lambda0)
        dt = time_window / N
        t = collect((-N/2:N/2-1) .* dt)
        
        # Frequency domain
        dW = 2π / time_window
        W = fftshift(fftfreq(N, 1 / dt)) .* 2π
        W = collect((-N/2:N/2-1) .* dW)
        
        omega0 = 2π * (299792458.0 / lambda0) # c / lambda0
        return Grid(N, dt, t, W, omega0)
    end

    struct Pulse
        grid::Grid
        At::Vector{ComplexF64} # Time-domain pulse
        AW::Vector{ComplexF64} # Frequency-domain pulse
    end

    function sech_pulse(grid::Grid, P0::Real, T0::Real)
        tau = T0 / 1.7627
        At = sqrt(P0) .* sech.(grid.t ./ tau)
        AW = fft(At) ./ grid.N # Normalize FFT
        return Pulse(grid, At, AW)
    end

    struct TaylorDispersion
        betas::Vector{Float64}
    end

    struct Medium
        length::Float64
        gamma::Float64
        loss::Float64
        dispersion::TaylorDispersion
        lambda0::Float64

        function Medium(; length, gamma, loss, dispersion, lambda0)
            new(length, gamma, loss, dispersion, lambda0)
        end
    end

    struct SimParams
        medium::Medium
        z_saves::Int
        raman_model
        self_steepening::Bool
        rtol::Float64
        atol::Float64
    end

    struct ConstantGamma
        gamma::Float64
    end

    struct GNLSEProblem
        medium::Medium
        grid::Grid
        initial_pulse::Pulse
        sim_params::SimParams
        gamma_coefficient::ConstantGamma
    end

    struct Solution
        t::Vector{Float64}
        W::Vector{Float64}
        omega0::Float64
        Z::Vector{Float64} # Propagation distances
        At::Matrix{ComplexF64} # Time domain pulse evolution
        AW::Matrix{ComplexF64} # Frequency domain pulse evolution
    end

    # Mock solve function for Fiber step: it just propagates a bit and adds some dummy change
    function solve(problem::GNLSEProblem; progress::Bool=true)
        # Simulate a small propagation. For testing, we just update the pulse directly.
        # In a real scenario, this would involve a complex solver.
        
        # Dummy propagation: apply some minimal change to energy and shift phase
        final_At = deepcopy(problem.initial_pulse.At)
        final_AW = deepcopy(problem.initial_pulse.AW)
        
        # Simulate some minor loss and phase shift to show "propagation"
        loss_factor_dummy = 0.99
        phase_shift_dummy = exp.(1im .* problem.grid.t .* 1e9) # Arbitrary phase shift
        
        final_At .*= sqrt(loss_factor_dummy) .* phase_shift_dummy
        final_AW .*= sqrt(loss_factor_dummy) .* fft(phase_shift_dummy) ./ problem.grid.N # Adjust AW accordingly

        z_points = [0.0, problem.medium.length]
        
        return Solution(
            problem.grid.t,
            problem.grid.W,
            problem.grid.omega0,
            z_points,
            hcat(problem.initial_pulse.At, final_At),
            hcat(problem.initial_pulse.AW, final_AW),
        )
    end
    
    # Mock pulse_energy function
    function pulse_energy(pulse::Pulse)
        return sum(abs2.(pulse.At)) * pulse.grid.dt
    end
end

# Now we can import the PropagationSteps module and use our mock types
using .MockJuGNLSE
using JuGNLSE.PropagationSteps # Import the actual module to test

@testset "PropagationSteps" begin
    # Setup for tests
    N = 2^10
    time_window = 10e-12
    lambda0 = 1550e-9
    grid = MockJuGNLSE.create_grid(N, time_window, lambda0)

    P0 = 1000.0 # Peak power in Watts
    T0 = 50e-15 # Pulse duration in seconds
    initial_pulse = MockJuGNLSE.sech_pulse(grid, P0, T0)
    initial_energy = MockJuGNLSE.pulse_energy(initial_pulse)

    # Test 1: Loss step
    @testset "Loss Step" begin
        pulse = deepcopy(initial_pulse)
        loss_dB = 3.0 # 3 dB loss means energy halves
        loss_step = Loss(loss_dB)
        propagate!(pulse, loss_step)
        
        final_energy = MockJuGNLSE.pulse_energy(pulse)
        expected_energy = initial_energy * 10^(-loss_dB / 10)
        @test final_energy ≈ expected_energy rtol=1e-6
    end

    # Test 2: Amplifier step
    @testset "Amplifier Step" begin
        pulse = deepcopy(initial_pulse)
        gain_dB = 3.0 # 3 dB gain means energy doubles
        amplifier_step = Amplifier(gain_dB)
        propagate!(pulse, amplifier_step)
        
        final_energy = MockJuGNLSE.pulse_energy(pulse)
        expected_energy = initial_energy * 10^(gain_dB / 10)
        @test final_energy ≈ expected_energy rtol=1e-6
    end

    # Test 3: Fiber step (minimal test, focusing on if it runs without error and pulse is updated)
    @testset "Fiber Step" begin
        pulse = deepcopy(initial_pulse)
        
        # Create a mock medium for the Fiber step
        mock_dispersion = MockJuGNLSE.TaylorDispersion([-11.83e-27])
        mock_medium = MockJuGNLSE.Medium(
            length=0.1, gamma=0.11, loss=0.0, dispersion=mock_dispersion, lambda0=lambda0
        )
        
        fiber_length = 0.1 # meters
        fiber_step = Fiber(mock_medium, fiber_length)
        
        initial_At_sum = sum(abs2.(pulse.At))
        
        # Propagate through fiber
        sol = propagate!(pulse, fiber_step; progress=false)
        
        # Check if pulse fields are updated (mock solve changes them slightly)
        @test !(sum(abs2.(pulse.At)) ≈ initial_At_sum) # Energy should change
        @test size(sol.At, 2) == 2 # Initial + final point from mock solve
        @test sol.Z[end] ≈ fiber_length # Check if length is correctly recorded
    end

    # Test 4: Composition of steps (Fiber -> Loss -> Amplifier)
    @testset "Composite Pipeline" begin
        pulse = deepcopy(initial_pulse)
        
        # Mock medium for Fiber
        mock_dispersion = MockJuGNLSE.TaylorDispersion([-11.83e-27])
        mock_medium = MockJuGNLSE.Medium(
            length=0.1, gamma=0.11, loss=0.0, dispersion=mock_dispersion, lambda0=lambda0
        )
        
        fiber_step = Fiber(mock_medium, 0.1)
        loss_step = Loss(3.0) # -3 dB
        amplifier_step = Amplifier(6.0) # +6 dB

        # Initial energy before pipeline
        current_energy = MockJuGNLSE.pulse_energy(pulse)
        
        # Expected energy after loss: initial_energy * 10^(-3/10)
        expected_energy_after_loss = current_energy * 10^(-3/10)
        
        # Expected energy after amplifier: expected_energy_after_loss * 10^(6/10)
        expected_final_energy = expected_energy_after_loss * 10^(6/10)
        
        # The mock Fiber step also applies a dummy loss (0.99 factor)
        # So we need to account for that in the expected final energy
        expected_final_energy_with_fiber_mock = initial_energy * 0.99 * 10^(-3/10) * 10^(6/10)


        pipeline = [fiber_step, loss_step, amplifier_step]
        sol = propagate!(pulse, pipeline; progress=false)
        
        final_energy = MockJuGNLSE.pulse_energy(pulse)
        
        # Due to the mock solve introducing a 0.99 factor in the fiber step,
        # we adjust the expected value.
        @test final_energy ≈ expected_final_energy_with_fiber_mock rtol=1e-6

        # Verify that the solution object contains the Fiber step's evolution
        @test size(sol.At, 2) == 2 # Only fiber step adds to solution history
        @test sol.Z[end] ≈ 0.1 # Length of the fiber step
    end
    
    # Test 5: Filter step (minimal test, just ensuring it runs and AW/At are modified)
    @testset "Filter Step" begin
        pulse = deepcopy(initial_pulse)
        
        # A dummy filter function that zeroes out high frequencies
        function dummy_filter(W, AW)
            filtered_AW = deepcopy(AW)
            # Example: Zero out frequencies above a certain threshold
            threshold_W = 2π * 100e9 # 100 GHz
            filtered_AW[abs.(W) .> threshold_W] .= 0.0
            return filtered_AW
        end

        filter_step = Filter(dummy_filter)
        
        initial_AW_sum = sum(abs2.(pulse.AW))
        propagate!(pulse, filter_step)
        final_AW_sum = sum(abs2.(pulse.AW))
        
        # Expect AW to change due to filtering (and thus energy)
        @test !(final_AW_sum ≈ initial_AW_sum)
        # Also check if At is updated from AW
        @test sum(abs2.(ifft(pulse.AW) .* pulse.grid.N)) ≈ sum(abs2.(pulse.At)) rtol=1e-6
    end
end
