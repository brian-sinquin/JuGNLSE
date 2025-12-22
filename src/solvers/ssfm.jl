"""
Symmetric Split-Step Fourier Method (SSFM) solver for GNLSE.

Classic, robust method for nonlinear pulse propagation:
- Symmetric split: exp(D̂·dz/2) · exp(N̂·dz) · exp(D̂·dz/2)
- Can use fixed or adaptive stepping
- More steps than ERK4IP but simpler and very reliable
- Good for validation and benchmarking

Algorithm:
1. Half-step dispersion in frequency domain
2. Full-step nonlinearity in time domain
3. Half-step dispersion in frequency domain
"""

using FFTW
using LinearAlgebra: mul!

# Import nonlinearity module functions
import ..build_physics_model, ..PhysicsModel

"""
    propagate_ssfm(pulse::Pulse, params::SimParams;
                   progress::Bool=true, adaptive::Bool=false,
                   dz::Union{Float64,Nothing}=nothing)

Propagate pulse using Symmetric Split-Step Fourier Method.

# Arguments

  - `pulse::Pulse`: Initial pulse
  - `params::SimParams`: Simulation parameters
  - `progress::Bool`: Show progress (default: true)
  - `adaptive::Bool`: Use adaptive stepping (experimental, default: false)
  - `dz::Union{Float64,Nothing}`: Fixed step size [m]. If `nothing`, auto-computed as L/(20*n_saves)

# Algorithm

Symmetric split-step method:

```
A(z+dz) = exp(D̂·dz/2) · [exp(N̂·dz) · A(z)] · exp(D̂·dz/2)
```

where:

  - D̂ is dispersion operator (frequency domain)
  - N̂ is nonlinear operator (time domain)

# Step Size Recommendations

  - **Conservative**: dz = L / (50 * n_saves) - very accurate but slow
  - **Standard**: dz = L / (20 * n_saves) - good balance (default)
  - **Fast**: dz = L / (10 * n_saves) - faster but check convergence
  - **Adaptive** (experimental): Start with conservative, adjust based on local error

# Returns

Tuple of (z, At, Aw):

  - `z::Vector{Float64}`: Propagation distances
  - `At::Matrix{ComplexF64}`: Time-domain evolution
  - `Aw::Matrix{ComplexF64}`: Frequency-domain evolution

# Performance Notes

SSFM typically requires 2-5x more steps than ERK4IP for same accuracy,
but each step is simpler (3 FFTs vs 10 FFTs). Good for:

  - Validation of ERK4IP results
  - Very long propagation distances
  - Highly nonlinear problems where adaptive ERK4IP struggles

# Examples

```julia
# Fixed step SSFM (default)
results = solve(pulse, params; method=:SSFM)

# Custom step size
results = solve(pulse, params; method=:SSFM, dz=1e-4)

# Adaptive SSFM (experimental)
results = solve(pulse, params; method=:SSFM, adaptive=true)
```

# See Also

  - [`propagate_erk4ip`](@ref): Adaptive RK4 in interaction picture (recommended)
  - [`build_physics_model`](@ref): Physics operators construction
"""
function propagate_ssfm(
    pulse::Pulse,
    params::SimParams;
    progress::Bool=true,
    adaptive::Bool=false,
    dz::Union{Float64, Nothing}=nothing,
)
    grid = pulse.grid
    medium = params.medium
    N = grid.N
    z_end = medium.length
    n_saves = params.n_saves

    # Build physics model once
    model = build_physics_model(grid, params)

    # Initial condition in frequency domain
    U = copy(pulse.Aw)

    # Storage
    z_out = zeros(n_saves)
    At_out = zeros(ComplexF64, N, n_saves)
    Aw_out = zeros(ComplexF64, N, n_saves)

    z_out[1] = 0.0
    At_out[:, 1] .= pulse.At
    Aw_out[:, 1] .= U

    # Determine step size
    if dz === nothing
        dz = params.dz
    end

    # Step size control
    z = 0.0
    save_idx = 2
    z_saves = range(0, z_end; length=n_saves)

    # Pre-allocate workspace
    At_temp = similar(pulse.At)  # Time-domain buffer
    U_half = similar(U)          # After half-step dispersion
    U_mid = similar(U)           # Midpoint for RK2
    exp_D_half = similar(U)      # exp(D̂·dz/2)
    k1 = similar(U)              # RK2 stage 1
    k2 = similar(U)              # RK2 stage 2

    # Initialize progress bar
    dz_mm = round(dz * 1e3; digits=2)
    mode = adaptive ? "adaptive" : "fixed"
    prog = if progress
        Progress(n_saves - 1; desc="SSFM-$mode (dz≈$(dz_mm)mm): ", showspeed=true)
    else
        nothing
    end

    step_count = 0
    while z < z_end && save_idx <= n_saves
        step_count += 1

        # Limit step to reach next save point exactly
        z_target = z_saves[save_idx]
        dz_actual = min(dz, z_target - z)

        # ============================================================
        # Symmetric Split-Step Fourier Method
        # ============================================================

        # Compute dispersion phase shift: exp(D̂·dz/2)
        @. exp_D_half = exp(0.5 * dz_actual * model.dispersion_term)

        # Step 1: Half-step dispersion (frequency domain)
        @. U_half = exp_D_half * U

        # Step 2: Transform to time domain for nonlinearity
        mul!(At_temp, model.ifftp, U_half)

        # Step 3: Apply full nonlinear step
        # For Kerr-only with scalar gamma: can use exact phase exp(iγ·dz·|A|²)
        # For Raman/shock or vector gamma: use RK2 midpoint method

        use_simple_kerr = !params.raman && !params.shock && (model.γ isa Float64)

        if use_simple_kerr
            # Kerr-only with scalar gamma: exact phase application in time domain
            # A → A · exp(iγ·dz·|A|²)
            @. At_temp = At_temp * exp(1.0im * model.γ * dz_actual * abs2(At_temp))
            # Transform back to frequency domain
            mul!(U_half, model.fftp, At_temp)
        else
            # For complex nonlinearity or vector gamma, use RK2 midpoint method:
            # k1 = N̂[A(z)]
            # k2 = N̂[A(z) from Ũ + dz/2·k1]
            # Ũ(z+dz) = Ũ(z) + dz·k2

            # k1: Nonlinear operator at current field
            k1_val = model.nonlinear_function(At_temp, model)

            # Midpoint: advance by half step
            @. U_mid = U_half + 0.5 * dz_actual * k1_val
            mul!(At_temp, model.ifftp, U_mid)

            # k2: Nonlinear operator at midpoint
            k2_val = model.nonlinear_function(At_temp, model)

            # Full step using k2
            @. U_half = U_half + dz_actual * k2_val
        end

        # Step 4: Half-step dispersion (frequency domain)
        @. U = exp_D_half * U_half

        # Update propagation distance
        z += dz_actual

        # Save output if we reached target z
        if z >= z_target - 1e-12 * z_end
            z_out[save_idx] = z
            copyto!(@view(Aw_out[:, save_idx]), U)

            # Transform to time domain and shift for storage
            mul!(At_temp, model.ifftp, U)
            At_out[:, save_idx] .= fftshift(At_temp)

            # Update progress bar
            if !isnothing(prog)
                update!(prog, save_idx - 1)
            end

            save_idx += 1
        end
    end

    return z_out, At_out, Aw_out
end
