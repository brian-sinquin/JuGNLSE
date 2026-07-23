# JuGNLSE Architecture Expansion

## 1. Gamma Coefficient Design
To support wavelength and z-dependent gamma without breaking performance:

- **Type Hierarchy**:
  - `AbstractGammaCoefficient`
  - `ConstantGamma <: AbstractGammaCoefficient`
  - `WavelengthDependentGamma <: AbstractGammaCoefficient`
  - `ZDependentGamma <: AbstractGammaCoefficient`
  - `WavelengthAndZDependentGamma <: AbstractGammaCoefficient`

- **Interface**:
  - `get_gamma(g::AbstractGammaCoefficient, λ::Real, z::Real) -> Float64`
  - Uses Multiple Dispatch to call the appropriate method.

## 2. Solver Interface
- **Type Hierarchy**:
  - `AbstractGNLSESolver`
  - `RK4Solver <: AbstractGNLSESolver`
  - `AdaptiveStepSolver <: AbstractGNLSESolver`

- **Common Interface**:
  - `solve!(integrator::AbstractGNLSESolver, problem::GNLSEProblem, ...)`

## 3. GNLSEProblem Structure
Refactor the current `params` to a `GNLSEProblem` struct that holds the medium, the gamma coefficient model, and other physical parameters.
