# Changelog

All notable changes to Soliton.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-09-06

### Fixed

#### Critical Bug Fixes (from research-quality audit)

- **C2 — Pulse(Solution) spectral ordering**: Fixed missing `ifftshift` when constructing `Pulse` from `Solution`. The saved `AW` was in monotonic frequency order while `Pulse.AW` must be in FFT-natural order. This affected all scalar cascade and piping paths.

- **C3 — Filter spectral ordering**: Fixed `Filter` applying transfer function evaluated on monotonic `grid.W` directly to `pulse.AW` (FFT-natural order). Added `ifftshift` to the transfer vector so broadband filters correctly pass energy instead of suppressing it by ~100%.

- **C4 — Amplifying medium Kerr normalization**: Removed spurious extra `1/ω₀` factor in `_amplifying_spm` and `_amplifying_spm_raman`. Kerr term in zero-gain amplifier now matches passive medium exactly (verified by derivative norm ratio = 1).

- **H4 — Grid validation**: Added explicit rejection of `N < 2` and odd `N` in `create_grid`. Odd grids produce inconsistent `length(t) ≠ length(V)` and break FFT conventions.

### Added

- **13 regression tests** in `test_audit_fixes.jl` covering:
  - C2: Pulse(Solution) FFT invariant (`AW ≈ ifft(At)`)
  - C3: Broadband filter energy retention >99%
  - C4: Zero-gain amplifier Kerr derivative equals passive medium
  - H4: Grid validation for N=0,1,3,5,127 and even N≥2

### Changed

- Minimum Julia version unchanged (1.10)
- All 384 tests pass (371 existing + 13 new audit tests)

## [0.2.1] - 2026-08-XX

[Previous release notes...]