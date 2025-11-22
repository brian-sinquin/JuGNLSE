# JuGNLSE Development Notes

## Research Overview

This document contains notes from existing GNLSE implementations and research papers to guide the development of the Julia-based GNLSE solver.

---

## 1. Repository Analysis

### 1.1 xmhk/gnlse (Python, Original Implementation)
**Repository:** https://github.com/xmhk/gnlse  
**Language:** Python  
**Last Updated:** 2014  
**Status:** Educational/Historical reference

#### Key Features:
- Based on RK4IP MATLAB script from "Supercontinuum Generation in Optical Fibers" (Dudley & Taylor, 2010)
- Uses SciPy ODE solvers with adaptive step size
- Single-mode fiber simulation

#### Architecture:
1. **Core Functions:**
   - `prepare_sim_params()` - Initialize simulation parameters
   - `perform_simulation()` - Main integration loop
   - `GNLSE_RHS()` - Right-hand side of GNLSE equation

2. **Physical Effects:**
   - Raman response (3 models: Blow-Wood, Lin-Agrawal, Hollenbeck)
   - Self-steepening (shock term)
   - Frequency-dependent losses
   - Higher-order dispersion

3. **Raman Response Models:**
   - **Blow-Wood (1989)**: Single Lorentzian, fr = 0.18
   - **Lin-Agrawal (2006)**: Includes Boson peak, fr = 0.245
   - **Hollenbeck (2002)**: 13-component model, most accurate, fr ≈ 0.2

4. **Integration Methods:**
   - SciPy ODE solvers: 'dopri5' (default), 'dop853', 'lsoda', 'vode'
   - Adaptive step size with tolerances (reltol=1e-6, abstol=1e-9)

5. **Grid Setup:**
   - Time domain: N points (powers of 2 recommended)
   - Frequency domain: FFT-based
   - Dispersion: Taylor expansion or frequency vector

#### Example Demos:
- Raman shift (soliton self-frequency shift)
- Self-steepening effects
- Higher-order soliton propagation
- Supercontinuum generation via soliton fission
- Frequency-dependent losses

---

### 1.2 WUST-FOG/gnlse-python (Modern Python Package)
**Repository:** https://github.com/WUST-FOG/gnlse-python  
**Language:** Python 3.9+  
**Status:** Actively maintained (v2.0.1, Jan 2023)  
**Citation:** Redman et al., arXiv:2110.00298 (2021)

#### Key Improvements Over Original:
- **Modular design** with separate components
- **M-GNLSE**: Modified GNLSE accounting for mode profile dispersion
- Nonlinearity with frequency-dependent effective mode area
- Two dispersion operators (Taylor expansion vs effective refractive indices)
- Complete documentation at https://gnlse.readthedocs.io

#### Module Structure:
```
gnlse/
├── __init__.py
├── common.py              # Common utilities
├── dispersion.py          # Dispersion operators
├── envelopes.py           # Pulse shapes (Sech, Gaussian, CW, etc.)
├── gnlse.py              # Main solver class
├── import_export.py       # I/O functions (.mat files)
├── nonlinearity.py        # Nonlinear operators
├── raman_response.py      # Raman response functions
└── visualization.py       # Plotting utilities
```

#### Key Classes/Functions:
1. **Dispersion:**
   - Taylor expansion method
   - Effective refractive index interpolation
   
2. **Nonlinearity:**
   - Constant gamma (simple)
   - `NonlinearityFromEffectiveArea`: frequency-dependent Aeff
   
3. **Pulse Envelopes:**
   - Sech pulse
   - Gaussian pulse
   - Continuous wave (CW)
   - Custom envelopes

4. **Raman Response:**
   - Same 3 models as original
   - Normalized response functions

#### Example Simulations:
- `test_Dudley.py` - Supercontinuum with 3 pulse types
- `test_3rd_order_soliton.py` - Soliton evolution
- `test_dispersion.py` - Different dispersion operators
- `test_nonlinearity.py` - GNLSE vs M-GNLSE comparison
- `test_gvd.py` - Pulse broadening
- `test_raman.py` - Soliton fission with different Raman models
- `test_spm.py` - Self-phase modulation

#### Dependencies:
```
numpy
scipy
matplotlib
```

---

### 1.3 jtravs/SCGBookCode (MATLAB Reference)
**Repository:** https://github.com/jtravs/SCGBookCode  
**Language:** MATLAB  
**Status:** Reference implementation from book  
**Book:** "Supercontinuum Generation in Optical Fibers" (2010)

#### Structure:
- `gnlse.m` - Main GNLSE solver function (~70 lines)
- `test_Dudley.m` - Example driver script

#### Key Algorithm (RK4IP):
1. Define RHS of Eq. (3.13) from the book
2. Use MATLAB's `ode45` integrator
3. Interaction picture approach
4. Linear operator in frequency domain
5. Nonlinear operator in time domain

#### Notable Features:
- Very concise implementation (educational)
- Uses MATLAB's built-in ODE solvers
- Automatic error control
- Optional shock term (w0 parameter)
- Vectorized Raman convolution

---

### 1.4 WUST-FOG/cgnlse-python (Coupled GNLSE)
**Repository:** https://github.com/WUST-FOG/cgnlse-python  
**Language:** Python  
**Status:** Research code (2022)  
**Citations:**
- Stefańska et al., Opt. Lett. 47, 4183 (2022)
- Based on gnlse-python package

#### Purpose:
Simulate **birefringent microstructured fibers** with:
- Two coupled nonlinear Schrödinger equations
- Raman and Kerr nonlinearities
- High and low birefringence regimes

#### Physical Phenomena:
1. **Soliton trapping**
2. **Orthogonal Raman scattering**
3. **Vector modulation instability**
4. **Cross-polarization effects**

#### Key Simulations:
- `run_soliton_traping.py` - Soliton dynamics in birefringent fiber
- `run_modulation_instability.py` - Vector MI in normal dispersion regime

#### Module Structure:
```
cnlse/
├── (coupling implementation)
├── raman_polarisation
└── (extends gnlse-python)
```

---

## 2. Numerical Methods

### 2.1 Split-Step Fourier Method (SSFM)

**Basic Algorithm:**
```
For each step dz:
1. Linear step (frequency domain): A(ω) → A(ω) * exp(i*β(ω)*dz/2)
2. FFT to time domain: A(ω) → A(t)
3. Nonlinear step (time domain): A(t) → A(t) * exp(i*γ*|A(t)|²*dz)
4. FFT to frequency domain: A(t) → A(ω)
5. Linear step: A(ω) → A(ω) * exp(i*β(ω)*dz/2)
```

**Pros:**
- Fast (FFT-based)
- Easy to implement
- Good for weakly nonlinear regimes

**Cons:**
- Fixed step size (standard version)
- Second-order accurate (Strang splitting)
- Can be unstable for strong nonlinearity

### 2.2 RK4IP (Runge-Kutta 4th Order in Interaction Picture)

**Algorithm:**
```
Transform to interaction picture:
u(z) = A(z) * exp(-L*z)

RK4 on: du/dz = exp(-L*z) * N(exp(L*z) * u)

Where:
- L = linear operator (dispersion + loss)
- N = nonlinear operator
```

**Advantages:**
- 4th order accurate
- Adaptive step size possible
- Better for strong nonlinearity
- Handles broad bandwidth well

**Used by:**
- All three Python implementations
- MATLAB book code
- SciPy ODE solvers (dopri5, dop853)

---

## 3. Physics Implementation

### 3.1 GNLSE Equation

**Standard Form:**
```
∂A/∂z = -α/2*A - i*Σ(β_n/n! * ∂ⁿA/∂tⁿ) + i*γ*(1+iω₀⁻¹∂/∂t)*(A(t)∫R(t')|A(t-t')|²dt')
```

**Components:**
1. **Loss term:** α/2
2. **Dispersion:** β₂, β₃, β₄, ... (Taylor expansion)
3. **Nonlinearity:** γ (Kerr coefficient)
4. **Shock (self-steepening):** ω₀⁻¹∂/∂t term
5. **Raman response:** R(t) = (1-fᵣ)δ(t) + fᵣ*hᵣ(t)

### 3.2 Parameter Units

**Typical Values:**
- Center wavelength: λ₀ = 800-1550 nm
- Pulse duration: FWHM = 10-100 fs
- Fiber length: L = 0.01-10 m
- Nonlinear coefficient: γ = 0.001-0.1 W⁻¹m⁻¹
- GVD: β₂ = -50 to +50 ps²/km
- Loss: α = 0.1-10 dB/km
- Raman fraction: fᵣ = 0.18-0.245

### 3.3 Dispersion Models

**Method 1: Taylor Expansion**
```julia
β(ω) = Σ βₙ/n! * (ω - ω₀)ⁿ
```

**Method 2: Sellmeier/Refractive Index**
```julia
β(ω) = n(ω) * ω / c
```
- More accurate for broad bandwidth
- Requires interpolation

### 3.4 Raman Response Functions

**1. Blow-Wood (1989):**
```julia
h_R(t) = (τ₁² + τ₂²)/(τ₁*τ₂²) * exp(-t/τ₂) * sin(t/τ₁)
τ₁ = 12.2 fs, τ₂ = 32.0 fs
fᵣ = 0.18
```

**2. Lin-Agrawal (2006):**
```julia
h_R(t) = fₐ*hₐ(t) + f_b*h_b(t) + f_c*hₐ(t)
# Includes Boson peak
fᵣ = 0.245
```

**3. Hollenbeck-Cantrell (2002):**
```julia
# 13 Lorentzian oscillators
# Most accurate experimental fit
fᵣ ≈ 0.20
```

---

## 4. Software Design Recommendations

### 4.1 Module Structure (Proposed for JuGNLSE)

```julia
JuGNLSE/
├── src/
│   ├── JuGNLSE.jl           # Main module
│   ├── core/
│   │   ├── solver.jl        # Main GNLSE solver
│   │   ├── grid.jl          # Time/frequency grid setup
│   │   └── integration.jl   # RK4IP, SSFM methods
│   ├── physics/
│   │   ├── dispersion.jl    # Dispersion operators
│   │   ├── nonlinearity.jl  # Nonlinear operators
│   │   ├── raman.jl         # Raman response models
│   │   └── loss.jl          # Loss/gain
│   ├── pulses/
│   │   └── envelopes.jl     # Pulse shapes
│   ├── fiber/
│   │   └── parameters.jl    # Fiber properties
│   └── utils/
│       ├── io.jl            # Input/output
│       └── plotting.jl      # Visualization
└── test/
    ├── test_soliton.jl
    ├── test_spm.jl
    └── test_supercontinuum.jl
```

### 4.2 Key Julia Packages to Use

**Required:**
- `FFTW.jl` - Fast Fourier transforms (already added)
- `DifferentialEquations.jl` - ODE solvers (RK4IP)
- `LinearAlgebra.jl` - Matrix operations

**Recommended:**
- `Interpolations.jl` - For dispersion curves
- `StaticArrays.jl` - Performance optimization
- `Parameters.jl` or `@with_kw` - Parameter structures
- `Plots.jl` or `Makie.jl` - Visualization
- `HDF5.jl` or `JLD2.jl` - Data storage
- `Unitful.jl` - Physical units (optional)
- `BenchmarkTools.jl` - Performance testing

**Optional (Advanced):**
- `CUDA.jl` - GPU acceleration
- `LoopVectorization.jl` - SIMD optimization
- `Enzyme.jl` - Automatic differentiation (for optimization)

### 4.3 Type System Design

```julia
# Fiber parameters
struct FiberParams{T<:Real}
    length::T              # m
    gamma::T               # W⁻¹m⁻¹
    betas::Vector{T}       # ps^n/km
    alpha::T               # dB/km (or vector)
    lambda0::T             # m
end

# Raman response
abstract type RamanModel end
struct BlowWood <: RamanModel end
struct LinAgrawal <: RamanModel end
struct Hollenbeck <: RamanModel end

# Pulse envelope
struct Pulse{T<:Complex}
    At::Vector{T}          # Time domain
    Aw::Vector{T}          # Frequency domain
    t::Vector{Float64}     # Time grid
    omega::Vector{Float64} # Frequency grid
end

# Simulation parameters
struct SimParams
    fiber::FiberParams
    N::Int                 # Grid points
    n_saves::Int           # Output points
    raman::Bool
    shock::Bool
    raman_model::RamanModel
    fr::Float64           # Raman fraction
end
```

### 4.4 API Design

**High-level Interface:**
```julia
using JuGNLSE

# Setup fiber
fiber = Fiber(
    length = 0.15,          # 15 cm
    gamma = 0.11,           # W⁻¹m⁻¹
    betas = [0, 0, -11.83e-27, 8.13e-41],  # ps^n/m
    lambda0 = 835e-9        # m
)

# Create pulse
pulse = SechPulse(
    FWHM = 50e-15,         # 50 fs
    power_peak = 10000,     # W
    lambda0 = 835e-9        # m
)

# Setup simulation
sim = Simulation(
    fiber = fiber,
    N = 2^12,              # 4096 points
    n_saves = 200,
    raman = true,
    raman_model = Hollenbeck(),
    shock = true
)

# Run simulation
results = solve(sim, pulse)

# Plot results
plot_evolution(results, domain=:both)
plot_spectrum(results, z=[0.0, fiber.length])
```

**Low-level Interface:**
```julia
# Manual setup for advanced users
grid = Grid(N=2^13, center_wavelength=1.55e-6, time_window=10e-12)
fiber = FiberParams(...)
pulse = create_pulse(grid, ...)

# Custom integration
integrator = RK4IP(grid, fiber, reltol=1e-7, abstol=1e-10)
z, At, Aw = propagate(integrator, pulse, fiber.length, n_saves=200)
```

---

## 5. Performance Considerations

### 5.1 Julia Advantages
- **JIT compilation**: Near C/Fortran speed
- **Multiple dispatch**: Clean physics separation
- **SIMD/parallelization**: Built-in
- **GPU support**: CUDA.jl for massive speedup
- **Type stability**: Critical for performance

### 5.2 Optimization Strategies

1. **Type Stability:**
   ```julia
   # Good
   function gnlse_rhs(Aw::Vector{ComplexF64}, params::SimParams)::Vector{ComplexF64}
   
   # Bad
   function gnlse_rhs(Aw, params)  # Type inference issues
   ```

2. **Pre-allocate Arrays:**
   ```julia
   # Pre-allocate output arrays
   At_out = similar(At)
   fft_plan = plan_fft(At)  # Create once, reuse
   ```

3. **In-place Operations:**
   ```julia
   # Good
   @. Aw_out = Aw * exp(linop * z)
   
   # Bad
   Aw_out = Aw .* exp.(linop .* z)  # Multiple allocations
   ```

4. **Use Views:**
   ```julia
   # Good
   @views result[:, i] = compute(data[:, i])
   
   # Bad
   result[:, i] = compute(data[:, i])  # Copy
   ```

### 5.3 Benchmarking Targets

**Reference (gnlse-python):**
- 4096 points, 200 z-steps
- ~10-30 seconds on modern CPU

**Julia Goal:**
- 5-10x faster than Python
- 1-3 seconds for same problem
- GPU: Sub-second for moderate sizes

---

## 6. Testing Strategy

### 6.1 Unit Tests

```julia
@testset "Dispersion" begin
    # Test Taylor expansion
    # Test interpolation
    # Test energy conservation
end

@testset "Raman Response" begin
    # Test normalization
    # Compare 3 models
    # Test frequency domain transform
end

@testset "Nonlinearity" begin
    # Test SPM phase
    # Test self-steepening
end
```

### 6.2 Integration Tests

**Standard Test Cases:**
1. **1st Order Soliton:** N=1, should propagate unchanged
2. **Higher-Order Soliton:** N=3, periodic evolution
3. **GVD Pulse Broadening:** Linear regime
4. **SPM:** Spectral broadening without dispersion
5. **Soliton Self-Frequency Shift:** Raman effect
6. **Supercontinuum:** Complex nonlinear dynamics

### 6.3 Validation Against Literature

**Compare with:**
- Dudley et al., RMP 78, 1135 (2006) - Fig. 3
- Agrawal's book examples
- gnlse-python test cases
- SCGBookCode MATLAB results

---

## 7. Documentation Plan

### 7.1 User Guide
- Installation instructions
- Quick start tutorial
- Examples gallery
- Parameter reference
- FAQ

### 7.2 Developer Guide
- Architecture overview
- Contributing guidelines
- Testing procedures
- Performance tips

### 7.3 API Reference
- Automatic from docstrings
- Use Documenter.jl

### 7.4 Examples
- Basic pulse propagation
- Soliton dynamics
- Supercontinuum generation
- Raman effects
- Custom fiber design
- Optimization workflows

---

## 8. Related Scientific Papers

### Key References:

1. **Dudley et al., RMP 78, 1135 (2006)**
   - Supercontinuum generation review
   - Standard test cases

2. **Agrawal, "Nonlinear Fiber Optics" (5th ed., 2012)**
   - Comprehensive GNLSE theory
   - Parameter values

3. **Blow & Wood, IEEE J. Quant. Elec. 25, 2665 (1989)**
   - Raman response model

4. **Lin & Agrawal, Opt. Lett. 31, 3086 (2006)**
   - Improved Raman model with Boson peak

5. **Hollenbeck & Cantrell, JOSA B 19, (2002)**
   - Accurate multi-mode Raman response

6. **Hult, J. Lightwave Tech. 25, 3770 (2007)**
   - RK4IP method details

7. **Redman et al., arXiv:2110.00298 (2021)**
   - gnlse-python software paper

8. **Stefańska et al., Opt. Lett. 47, 4183 (2022)**
   - Coupled GNLSE applications

---

## 9. Future Extensions

### 9.1 Core Features (v1.0)
- [x] Basic GNLSE solver
- [ ] RK4IP integration
- [ ] Three Raman models
- [ ] Self-steepening
- [ ] Multiple pulse shapes
- [ ] Basic visualization
- [ ] Standard test suite

### 9.2 Advanced Features (v2.0)
- [ ] Adaptive step size
- [ ] Coupled GNLSE (birefringence)
- [ ] Multi-mode fibers
- [ ] GPU acceleration
- [ ] Parallel ensemble runs
- [ ] Advanced dispersion models
- [ ] Saturable absorption

### 9.3 Optimization Features (v3.0)
- [ ] Automatic differentiation
- [ ] Parameter optimization
- [ ] Inverse design
- [ ] Machine learning integration
- [ ] Uncertainty quantification

---

## 10. Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
- [x] Package setup with PkgTemplates
- [x] Add FFTW dependency
- [ ] Grid generation
- [ ] Basic types and structures
- [ ] Unit tests framework

### Phase 2: Physics Implementation (Weeks 3-4)
- [ ] Linear operator (dispersion)
- [ ] Nonlinear operator
- [ ] Raman response functions
- [ ] Loss/gain terms
- [ ] Pulse envelopes

### Phase 3: Solver (Weeks 5-6)
- [ ] SSFM implementation
- [ ] RK4IP with DifferentialEquations.jl
- [ ] Adaptive stepping
- [ ] Energy conservation checks

### Phase 4: Validation (Week 7)
- [ ] Compare with gnlse-python
- [ ] Compare with MATLAB code
- [ ] Standard test cases
- [ ] Convergence studies

### Phase 5: Documentation (Week 8)
- [ ] API documentation
- [ ] User guide
- [ ] Example notebooks
- [ ] Benchmarks

### Phase 6: Optimization (Week 9-10)
- [ ] Performance profiling
- [ ] Memory optimization
- [ ] SIMD/Threading
- [ ] (Optional) GPU version

---

## 11. Notes from Code Review

### Python Implementation Insights:

1. **FFT Conventions:**
   - Use `fftshift` for proper frequency ordering
   - Energy scaling: `scalefak = sqrt(dt/dom * N)`
   - Parseval's theorem verification important

2. **Raman Convolution:**
   - Compute in frequency domain for efficiency
   - FFT(h_R) computed once and cached
   - Response normalized by area

3. **Integration Picture:**
   - Transform: `u = A * exp(-L*z)` at each step
   - Solves: `du/dz = exp(-L*z) * N(exp(L*z)*u)`
   - More stable than direct approach

4. **Status Reporting:**
   - Progress updates important for long simulations
   - Estimated time remaining helpful

5. **I/O:**
   - MATLAB .mat format popular for compatibility
   - Save both time and frequency fields
   - Include simulation parameters in output

### MATLAB Implementation Insights:

1. **Compact Code:**
   - Entire solver ~70 lines
   - Inline RHS function definition
   - Leverages MATLAB's ode45 heavily

2. **Error Control:**
   - RelTol = 1e-5
   - AbsTol = 1e-12
   - NormControl = 'on'

3. **Callback Functions:**
   - Progress reporting via OutputFcn
   - Can add custom callbacks for analysis

---

## 12. Questions to Address

1. **Should we use DifferentialEquations.jl or custom RK4IP?**
   - Pro DE.jl: Mature, many solvers, adaptive
   - Pro custom: More control, potential optimization
   - **Decision:** Start with custom, add DE.jl option later

2. **In-place vs allocating operations?**
   - In-place faster but more complex API
   - **Decision:** In-place for core, allocating for user API

3. **Unit system?**
   - SI units only (simple)
   - Unitful.jl (type-safe but overhead)
   - **Decision:** SI units with clear documentation

4. **GPU support priority?**
   - CUDA.jl works well with FFTs
   - **Decision:** CPU first, GPU as v2.0 feature

5. **Visualization?**
   - Plots.jl (widespread) or Makie.jl (faster, prettier)
   - **Decision:** Support both via recipes

---

## 13. Code Snippets to Reference

### Python FFT Pattern:
```python
# Frequency domain (shifted)
Aw = np.fft.ifft(At)

# Apply linear operator
Aw_prop = Aw * np.exp(linop * z)

# Back to time domain
At = np.fft.fft(Aw_prop)
```

### Julia Translation:
```julia
using FFTW

# Frequency domain
Aw = ifft(At)

# Apply linear operator
@. Aw_prop = Aw * exp(linop * z)

# Back to time domain
At = fft(Aw_prop)
```

### Raman Convolution Pattern:
```julia
# Pre-compute Raman in frequency domain
RW = N * ifft(fftshift(h_R))

# During propagation
IT = abs2.(At)  # Intensity
RS = dt * fr * fft(ifft(IT) .* RW)  # Raman term
```

---

## 14. Performance Notes

**Python bottlenecks:**
- List comprehension in output
- Repeated FFT planning
- No native SIMD

**Julia advantages:**
- FFT plans cached
- SIMD automatic
- Type specialization
- Stack allocation of small arrays

**Expected speedup:**
- Compute-bound: 5-10x
- Memory-bound: 2-5x
- Overall: ~5x faster than Python

---

## End of Research Notes

**Last Updated:** 2025-11-22  
**Status:** Ready for implementation  
**Next Step:** Begin Phase 1 (Core Infrastructure)
