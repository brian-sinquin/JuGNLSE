# Changelog

All notable changes to Soliton.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-09-07

### Fixed

- Preserve FFT ordering when converting scalar and vectorial `Solution` objects
  back to pulses, including when frequency-domain output was not saved.
- Apply scalar and vectorial filter transfer functions in the same FFT ordering
  as pulse spectra.
- Correct Kerr normalization in amplifying media and keep saturated gain
  independent of the nonlinear spectral coefficient, with and without Raman.
- Compute photon number correctly from time-domain output when frequency-domain
  output was not saved.
- Normalize spectral ordering in pulse and solution coherence calculations, and
  support solutions without saved frequency-domain output.
- Use the correct detuning-frequency axis for SHG FROG traces.
- Reject grids with fewer than two points or an odd number of points, which are
  incompatible with the package's FFT grid conventions.

### Changed

- Clarified the frequency-domain storage conventions for scalar and vectorial
  pulses and the input convention for filter transfer functions.
- Expanded regression coverage for cascades, filters, nonlinear gain, Raman,
  self-steepening, spectral analysis, and FFT edge cases.

## [0.2.1] - 2026-08-03

### Changed

- Renamed the package and Julia module from GNLSE.jl to Soliton.jl while
  preserving the package UUID.
- Updated documentation, tests, workflows, and precompiled sysimage artifact
  names for the Soliton.jl name.
- Updated GitHub Actions dependencies used by the release workflows.

### Added

- Added the Zenodo DOI badge and citation metadata.

## [0.2.0] - 2026-08-02

### Changed

- Renamed JuGNLSE.jl to GNLSE.jl for Julia General registry compatibility.

[0.2.2]: https://github.com/brian-sinquin/Soliton.jl/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/brian-sinquin/Soliton.jl/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/brian-sinquin/Soliton.jl/releases/tag/v0.2.0
