```julia
using JuGNLSE
using FFTW

# 1. Define the simulation grid
grid = create_grid(2^13, 12.5e-12, 835e-9) # N, time_window [s], lambda0 [m]

# 2. Define a base medium (e.g., for fiber properties)
medium = Medium(
    length=1.0, # This length is a placeholder for the medium definition, actual propagation length is in Fiber step
    gamma=0.11, # 1/(W*m)
    loss=0.0, # dB/m
    betas=[-11.83e-27, 8.15e-41, -1.33e-54], # Taylor series dispersion coefficients
    lambda0=835e-9
)

# 3. Create an initial pulse
Pmax = 1000.0 # Peak power [W]
FWHM = 50e-15 # Full-width at half-maximum [s]
pulse = sech_pulse(grid, Pmax, FWHM)

# 4. Define propagation steps
fiber1 = Fiber(medium, 0.1, 100) # 0.1m fiber, 100 save points
loss_element = Loss(3.0) # 3 dB loss
fiber2 = Fiber(medium, 0.2, 200) # 0.2m fiber, 200 save points

# Example filter function (simple bandpass filter)
function bandpass_filter(W, AW; center_frequency=grid.omega0, bandwidth=2e13) # 20 THz bandwidth
    # Gaussian filter in frequency domain
    filter_shape = exp.(-(W .- center_frequency).^2 ./ (2 * (bandwidth/2.355)^2))
    return AW .* filter_shape
end
filter_step = Filter(bandpass_filter)

amplifier_step = Amplifier(10.0) # 10 dB gain

# 5. Create a pipeline of steps
pipeline = [fiber1, loss_element, filter_step, amplifier_step, fiber2]

# 6. Propagate the pulse through the pipeline
println("Starting pipeline propagation...")
solution = propagate!(pulse, pipeline; progress=true)
println("Pipeline propagation finished.")

# 7. Access results
println("\n--- Simulation Results ---")
println("Final pulse peak power: ", round(peak_power(pulse); sigdigits=4), " W")
println("Total propagation distance: ", solution.Z[end], " m")
println("Number of saved points: ", length(solution.Z))

# Further analysis and plotting can be done with the 'solution' object
# using functions from JuGNLSE.analysis module.
```
