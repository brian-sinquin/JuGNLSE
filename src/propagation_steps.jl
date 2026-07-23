
module PropagationSteps

using ..JuGNLSE: Pulse, Grid, Medium, SimParams, GNLSEProblem, ConstantGamma, Solution, TaylorDispersion, solve
# using ..Solvers: propagate_erk4ip # Assuming this is the internal function used by solve

export AbstractPropagationStep, Fiber, Loss, Filter, Amplifier, propagate!

"""
    AbstractPropagationStep

Abstract base type for all propagation steps in a JuGNLSE pipeline.
Concrete subtypes define specific operations like fiber propagation, loss, filtering, or amplification.
"""
abstract type AbstractPropagationStep end

"""
    Fiber(medium::Medium, length::Real, z_saves::Int=200)

Represents a segment of optical fiber through which a pulse will propagate.

# Fields
  - `medium::Medium`: The optical medium parameters for this fiber segment.
  - `length::Float64`: The physical length of this fiber segment in meters.
  - `z_saves::Int`: The number of points along the fiber at which to save the pulse state.

# Arguments
  - `medium`: A `Medium` object defining the fiber's properties (dispersion, nonlinearity, loss).
  - `length`: The length of this specific fiber segment in meters. Must be positive.
  - `z_saves`: The number of intermediate points where the pulse state will be saved during propagation through this segment. Must be positive. Defaults to 200.

# Notes
This constructor creates a new `Medium` instance internally, ensuring that the `length` field of the `Medium` matches the `length` of this `Fiber` segment, even if the input `medium` had a different overall length.
"""
struct Fiber <: AbstractPropagationStep
    medium::Medium
    length::Float64 # Length of the fiber segment
    z_saves::Int # Number of save points within this fiber segment

    function Fiber(medium::Medium, length::Real, z_saves::Int=200)
        length > 0 || throw(ArgumentError("Fiber length must be positive"))
        z_saves > 0 || throw(ArgumentError("z_saves must be positive"))
        # Create a new medium with the specified length for this segment
        segment_medium = Medium(
            length=length,
            gamma=medium.gamma,
            loss=medium.loss,
            dispersion=medium.dispersion,
            lambda0=medium.lambda0,
        )
        new(segment_medium, Float64(length), z_saves)
    end
end

"""
    Loss(loss_dB::Real)

Represents a discrete loss element applied to the pulse energy.

# Fields
  - `loss_dB::Float64`: The amount of loss to apply in decibels (dB). Must be non-negative.

# Arguments
  - `loss_dB`: The loss value in dB. Must be non-negative.
"""
struct Loss <: AbstractPropagationStep
    loss_dB::Float64 # Loss in dB

    function Loss(loss_dB::Real)
        loss_dB >= 0 || throw(ArgumentError("Loss must be non-negative"))
        new(Float64(loss_dB))
    end
end

"""
    propagate!(pulse::Pulse, step::Fiber; rtol::Float64=1e-6, atol::Float64=1e-8, progress::Bool=true)

Propagates an optical `pulse` through a `Fiber` segment, solving the GNLSE.

# Arguments
  - `pulse::Pulse`: The optical pulse to propagate. It will be modified in-place with the final state after propagation through the `Fiber` segment.
  - `step::Fiber`: The `Fiber` step defining the medium and length for this propagation.
  - `rtol::Float64=1e-6`: Relative tolerance for the adaptive ODE solver.
  - `atol::Float64=1e-8`: Absolute tolerance for the adaptive ODE solver.
  - `progress::Bool=true`: If `true`, a progress bar will be displayed during propagation.

# Returns
  - `Solution`: A `Solution` object containing the pulse's evolution along the fiber segment.
"""
function propagate!(pulse::Pulse, step::Fiber; rtol::Float64=1e-6, atol::Float64=1e-8, progress::Bool=true)
    # Create simulation parameters for this step
    sim_params = SimParams(
        medium=step.medium,
        z_saves=step.z_saves,
        raman_model=nothing, # Or retrieve from a global/pulse setting if needed
        self_steepening=false, # Or retrieve from a global/pulse setting if needed
        rtol=rtol,
        atol=atol,
    )

    # Create a GNLSEProblem for this segment
    problem = GNLSEProblem(
        medium=step.medium,
        grid=pulse.grid,
        initial_pulse=pulse,
        sim_params=sim_params,
        gamma_coefficient=ConstantGamma(step.medium.gamma),
    )

    # Solve for this segment
    sol = solve(problem; progress=progress)

    # Update the pulse with the final state of this segment
    pulse.At .= sol.At[:, end]
    pulse.AW .= sol.AW[:, end]

    return sol
end

"""
    propagate!(pulse::Pulse, step::Loss)

Applies a discrete loss to the `pulse` energy based on the `Loss` step's `loss_dB` value.
The pulse's `At` and `AW` fields are modified in-place to reflect the energy reduction.

# Arguments
  - `pulse::Pulse`: The optical pulse to which the loss will be applied.
  - `step::Loss`: The `Loss` step containing the loss value in dB.

# Returns
`nothing`. This function modifies the `pulse` in-place and does not return a solution history.
"""
function propagate!(pulse::Pulse, step::Loss)
    loss_factor = 10^(-step.loss_dB / 10)
    pulse.At .*= sqrt(loss_factor)
    pulse.AW .*= sqrt(loss_factor)
    return nothing # Discrete loss doesn't return a solution history
end

"""
    Filter(filter_function::Function)

Represents an optical filter applied in the frequency domain.

# Fields
  - `filter_function::Function`: A function `f(W, AW)` that takes the absolute angular frequency grid `W` (Vector{Float64}) and the frequency domain pulse envelope `AW` (Vector{ComplexF64}) and returns the filtered `AW` (Vector{ComplexF64}).

# Arguments
  - `filter_function`: A function used to apply the filter. It should accept two arguments: the frequency grid `W` and the pulse's frequency domain data `AW`, and return the filtered `AW`.
"""
struct Filter <: AbstractPropagationStep
    filter_function::Function # A function f(W, AW) that returns filtered AW

    function Filter(filter_function::Function)
        new(filter_function)
    end
end

"""
    propagate!(pulse::Pulse, step::Filter)

Applies an optical filter to the `pulse` in the frequency domain using the provided `filter_function`.
The pulse's `AW` field is modified in-place with the filtered spectrum, and its `At` field is updated by inverse FFT.

# Arguments
  - `pulse::Pulse`: The optical pulse to which the filter will be applied.
  - `step::Filter`: The `Filter` step containing the filtering function.

# Returns
`nothing`. This function modifies the `pulse` in-place and does not return a solution history.
"""
function propagate!(pulse::Pulse, step::Filter)
    pulse.AW .= step.filter_function(pulse.grid.W, pulse.AW)
    pulse.At .= ifft(pulse.AW) .* pulse.grid.N # Update time domain from filtered frequency domain
    return nothing # Filter doesn't return a solution history
end

"""
    Amplifier(gain_dB::Real)

Represents an optical amplifier that increases pulse energy.

# Fields
  - `gain_dB::Float64`: The amount of gain to apply in decibels (dB). Must be non-negative.

# Arguments
  - `gain_dB`: The gain value in dB. Must be non-negative.
"""
struct Amplifier <: AbstractPropagationStep
    gain_dB::Float64 # Gain in dB

    function Amplifier(gain_dB::Real)
        new(Float64(gain_dB))
    end
end

"""
    propagate!(pulse::Pulse, step::Amplifier)

Applies optical amplification to the `pulse` energy based on the `Amplifier` step's `gain_dB` value.
The pulse's `At` and `AW` fields are modified in-place to reflect the energy increase.

# Arguments
  - `pulse::Pulse`: The optical pulse to which the amplification will be applied.
  - `step::Amplifier`: The `Amplifier` step containing the gain value in dB.

# Returns
`nothing`. This function modifies the `pulse` in-place and does not return a solution history.
"""
function propagate!(pulse::Pulse, step::Amplifier)
    gain_factor = 10^(step.gain_dB / 10)
    pulse.At .*= sqrt(gain_factor)
    pulse.AW .*= sqrt(gain_factor)
    return nothing # Amplifier doesn't return a solution history
end

"""
    propagate!(pulse::Pulse, steps::Vector{<:AbstractPropagationStep}; kwargs...)

Propagates an optical `pulse` through a sequence of `AbstractPropagationStep`s.
This function serves as a pipeline orchestrator, applying each step sequentially.

# Arguments
  - `pulse::Pulse`: The initial optical pulse to propagate through the pipeline. Its state will be updated in-place after each step.
  - `steps::Vector{<:AbstractPropagationStep}`: A vector of propagation steps to be applied in order.
  - `kwargs...`: Optional keyword arguments to pass to `propagate!` methods that support them (e.g., `rtol`, `atol`, `progress` for `Fiber` steps).

# Returns
  - `Solution`: A `Solution` object representing the pulse's evolution through all `Fiber` segments in the pipeline.
    If no `Fiber` steps are present, a dummy `Solution` representing the final pulse state at `Z=0.0` is returned.

# Notes
Discrete steps (like `Loss`, `Filter`, `Amplifier`) modify the `pulse` in-place but do not contribute to the returned `Solution` history, which only tracks propagation through `Fiber` segments.
"""
function propagate!(pulse::Pulse, steps::Vector{<:AbstractPropagationStep}; kwargs...)
    full_solution_Z = Float64[]
    full_solution_At = Matrix{ComplexF64}(undef, length(pulse.At), 0)
    full_solution_AW = Matrix{ComplexF64}(undef, length(pulse.AW), 0)

    # We create a scratch pulse to avoid modifying the input pulse unless necessary,
    # but we avoid deepcopy to keep memory allocations low.
    current_pulse = Pulse(copy(pulse.At), copy(pulse.AW), pulse.grid)

    for (i, step) in enumerate(steps)
        if step isa Fiber
            sol = propagate!(current_pulse, step; kwargs...)
            append!(full_solution_Z, sol.Z)
            full_solution_At = hcat(full_solution_At, sol.At)
            full_solution_AW = hcat(full_solution_AW, sol.AW)
        else
            # For discrete steps, we apply them and continue with the updated pulse
            propagate!(current_pulse, step)
            # No intermediate solution history for discrete steps, just update pulse
        end
    end

    # If no Fiber steps, return the final pulse state in a dummy solution (or decide on alternative)
    if isempty(full_solution_Z)
        # For now, just return the final pulse as a 1-step solution
        return Solution(
            current_pulse.grid.t,
            current_pulse.grid.W,
            current_pulse.grid.omega0,
            [0.0,], # dummy Z
            reshape(current_pulse.At, :, 1),
            reshape(current_pulse.AW, :, 1),
        )
    else
        # Reconstruct final solution object
        return Solution(
        current_pulse.grid.t,
        current_pulse.grid.W,
        current_pulse.grid.omega0,
        full_solution_Z,
        full_solution_At,
        full_solution_AW,
        )
    end # This end closes the if/else block
end # This end closes the propagate! function for pipelines

end # module PropagationSteps

