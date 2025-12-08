# FFT Convention Analysis for JuGNLSE

## Executive Summary

After analyzing the JuGNLSE codebase and comparing it with reference implementations (SCGBookCode MATLAB, gnlse-python, PyNLO), I have identified **critical sign and convention errors** in the FFT transformations and self-steepening implementation that cause self-steepening to arise in the wrong time domain direction.

## Current FFT Convention in JuGNLSE

The code currently uses an **INVERTED FFT CONVENTION**:
- `ifft(time_domain) → frequency_domain` 
- `fft(frequency_domain) → time_domain`

This is evident from:
1. **grid.jl** (line 58): Applies `fftshift` to omega after creation
2. **rk4ip.jl** (line 24): `At = fft_plan * du` (transforms frequency to time)
3. **rk4ip.jl** (line 32): `du .= ifft_plan * nonlin` (transforms time to frequency)
4. **nonlinearity.jl** (lines 50-52, 84): Comments explicitly state "inverted FFT"

## Standard FFT Convention (Reference Implementations)

All reference implementations (MATLAB gnlse.m, gnlse-python, PyNLO) use the **STANDARD FFT CONVENTION**:
- `fft(time_domain) → frequency_domain`
- `ifft(frequency_domain) → time_domain`

## Critical Issues Identified

### Issue 1: Derivative Sign in Self-Steepening (CRITICAL)

**Location**: `nonlinearity.jl` lines 50-52, 84

**Current Code**:
```julia
# Shock term: iγ/ω₀ * ∂|A|²/∂t
# With inverted FFT: ifft(time) → freq, fft(freq) → time
# Derivative: multiply by -iω (since ifft inverts the sign)
It_w = ifft_plan * It
@. It_w *= (-im * omega)
dI_dt = fft_plan * It_w
```

**Problem**: The derivative sign is **INCORRECT**. The comment "since ifft inverts the sign" is based on a misunderstanding of FFT conventions.

**Mathematical Reality**:
For time derivative ∂f/∂t:
- Standard FFT: `∂f/∂t = IFFT[+iω × FFT[f(t)]]`
- Inverted FFT: `∂f/∂t = FFT[+iω × IFFT[f(t)]]` (NOT -iω!)

The sign of the derivative operator iω is **invariant** under FFT convention changes. Only the transform direction changes, not the derivative operator itself.

**Consequence**: Self-steepening occurs in the **WRONG DIRECTION** in time domain, causing unphysical behavior.

**Correct Code Should Be**:
```julia
# Shock term: iγ/ω₀ * ∂|A|²/∂t
It_w = ifft_plan * It
@. It_w *= (im * omega)  # CORRECTED: +iω, not -iω
dI_dt = fft_plan * It_w
```

### Issue 2: fftshift Usage in Raman Response (MODERATE)

**Location**: `raman.jl` line 202

**Current Code**:
```julia
function raman_response_frequency(h_R::Vector{Float64}, grid::Grid)
    h_R_shifted = fftshift(h_R)
    RW = ifft(h_R_shifted)
    RW
end
```

**Problem**: The `fftshift` is applied **BEFORE** the FFT. This is inconsistent with the grid convention where omega is already fftshifted.

**Analysis**:
- `grid.omega` is already `fftshift`ed (grid.jl line 58)
- `h_R(t)` is defined on the time grid which is NOT shifted (centered at t=0)
- When computing FFT, we need h_R in the same order as the FFT expects

**Correct Approach**:
Since the grid.omega is fftshifted, and h_R is defined on the unshifted time grid (centered at t=0), we should:
1. NOT apply fftshift to h_R before FFT
2. OR apply ifftshift to h_R if it was meant to be in frequency order

**Issue**: The current approach may cause phase errors in Raman response, but this is less critical than the shock term.

### Issue 3: Grid Omega Already Shifted (LOW - Design Choice)

**Location**: `grid.jl` line 58

**Current Code**:
```julia
omega = fftshift(omega)
```

**Analysis**: This pre-applies fftshift to omega, which means:
- omega[1] corresponds to ω ≈ 0 (center frequency)
- omega array is in "human-readable" order: [low freq ... center ... high freq]
- But FFTW expects omega in FFT order: [0, +freq..., -freq...]

**Consequence**: 
- When multiplying by omega in frequency domain (e.g., for derivatives), the array order is correct
- BUT this creates confusion and may introduce errors when using fft/ifft

**Recommendation**: Either:
1. Keep omega unshifted (FFT order) and apply fftshift when needed for visualization only
2. OR consistently apply ifftshift before FFT operations and fftshift after

## Recommended Corrections

### Priority 1: Fix Self-Steepening Sign (CRITICAL)

**File**: `src/nonlinearity.jl`

**Lines to Change**: 51, 84, 258, 295

**Current**:
```julia
@. It_w *= (-im * omega)
```

**Corrected**:
```julia
@. It_w *= (im * omega)
```

**Justification**: The derivative operator ∂/∂t → +iω in frequency domain, regardless of FFT convention. The current -iω causes self-steepening in the wrong direction.

### Priority 2: Verify Raman Response Transform (MODERATE)

**File**: `src/raman.jl`

**Lines to Review**: 198-205

**Current Approach Issues**:
1. fftshift applied before ifft may cause phase errors
2. Scaling was removed (line 201 comment) which may affect normalization

**Recommended Test**:
Create a test that verifies Raman response convolution produces correct delay (Stokes shift to longer wavelengths).

### Priority 3: Standardize FFT Convention (LONG-TERM)

**Recommendation**: Migrate to **STANDARD FFT CONVENTION** for consistency with literature and other packages.

**Changes Required**:
1. **grid.jl**: Remove fftshift from omega initialization
2. **All solvers**: Swap fft ↔ ifft
3. **All nonlinearity.jl**: Update transform directions
4. **raman.jl**: Update Raman response frequency calculation

**Benefits**:
- Consistency with MATLAB gnlse, gnlse-python, PyNLO
- Easier to compare with published equations
- Reduced confusion about sign conventions

## Physical Tests to Validate Corrections

### Test 1: Self-Steepening Direction

**Setup**: 
- High-intensity pulse (N >> 1 soliton)
- Short duration (< 100 fs)
- Enable shock term only (no Raman, no dispersion)

**Expected Behavior**:
- Pulse should steepen on the **trailing edge** (positive time direction)
- Leading edge should remain smooth
- Shock wave formation at trailing edge

**Current Behavior** (with bug):
- Likely steepening on wrong edge or symmetric steepening

### Test 2: Raman-Induced Frequency Shift

**Setup**:
- Fundamental soliton (N=1)
- Enable Raman only
- Propagate several soliton periods

**Expected Behavior**:
- **Continuous red-shift** (Stokes shift to longer wavelengths)
- Self-frequency shift: Δω ∝ -z (negative means red)
- Soliton accelerates to longer wavelengths

**Current Behavior**:
- Should be checked if red-shift is occurring correctly

### Test 3: Combined Raman + Shock

**Setup**:
- Higher-order soliton (N=3-5)
- Enable both Raman and shock
- Fiber with anomalous dispersion

**Expected Behavior**:
- Soliton fission with proper temporal asymmetry
- Dispersive wave radiation
- Red-shifted solitons with asymmetric shape

### Test 4: Dispersive Wave Radiation

**Setup**:
- Perturbed fundamental soliton
- Strong higher-order dispersion (β₃, β₄)
- Enable all effects

**Expected Behavior**:
- Dispersive wave on **blue side** (shorter wavelengths)
- Phase-matched with soliton
- Specific frequency determined by phase-matching

## References

### Correct Mathematical Formulations

**Standard GNLSE** (Agrawal, "Nonlinear Fiber Optics", 6th ed):
```
∂A/∂z = -iβ₂/2 * ∂²A/∂t² + iβ₃/6 * ∂³A/∂t³ + ... 
        + iγ(1-f_R)|A|²A + iγf_R * A ∫h_R(t')|A(t-t')|²dt'
        + iγ/ω₀ * ∂/∂t[|A|²A]
```

**Frequency Domain Representation**:
```
∂Ã/∂z = +iβ(ω)Ã - α/2 Ã + N̂[A]
where β(ω) = Σ (βₙ/n!) * (ω-ω₀)ⁿ
```

**Self-Steepening Term**:
```
S(t) = iγ/ω₀ * ∂/∂t[(1-f_R)|A|² + f_R*(|A|² ⊗ h_R)] * A
     = iγ/ω₀ * [∂R/∂t * A + R * ∂A/∂t]
where R = (1-f_R)|A|² + f_R*(|A|² ⊗ h_R)
```

**Frequency Domain Derivative**:
```
∂f/∂t ↔ +iω * F(ω)  [STANDARD CONVENTION]
```

This is true because:
```
F[∂f/∂t] = ∫_{-∞}^{∞} ∂f/∂t * exp(-iωt) dt
         = [f*exp(-iωt)]_{-∞}^{∞} - ∫_{-∞}^{∞} f * (-iω) * exp(-iωt) dt
         = 0 + iω * ∫_{-∞}^{∞} f * exp(-iωt) dt
         = +iω * F(ω)
```

## Conclusion

The **primary issue** causing self-steepening to arise in the wrong direction is the **incorrect sign** of the derivative operator in the shock term implementation. Changing `-im * omega` to `+im * omega` in lines 51, 84, 258, and 295 of `nonlinearity.jl` will correct this critical bug.

Secondary issues with fftshift and FFT conventions should be addressed for long-term code health, but the shock term sign is the **immediate fix** required.
