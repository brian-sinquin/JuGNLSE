# Test Structure

This directory contains tests organized by type and purpose, following best practices from the comparative analysis of gnlse-python, PyNLO, and SCGBookCode.

## Directory Structure

```
test/
├── runtests.jl              # Main test runner
├── unit/                    # Unit tests - individual components
│   ├── test_types.jl        # Core types (Medium, Grid, Pulse)
│   ├── test_grid.jl         # Grid creation and properties
│   ├── test_pulse.jl        # Pulse initialization and manipulation
│   ├── test_dispersion.jl   # Dispersion operators
│   ├── test_nonlinearity.jl # Nonlinear operators
│   └── test_raman.jl        # Raman response models
├── integration/             # Integration tests - solvers and workflows
│   ├── test_ssfm.jl         # SSFM solver
│   ├── test_rk4ip.jl        # RK4IP solver
│   ├── test_erk4ip.jl       # ERK4IP solver (adaptive)
│   └── test_energy_conservation.jl
└── regression/              # Regression tests - reproduce publications
    ├── test_dudley2006.jl   # Dudley et al., RMP 78, 1135 (2006)
    └── test_solitons.jl     # Soliton propagation validation
```

## Test Philosophy

### Unit Tests
- **Purpose**: Test individual functions and components in isolation
- **Scope**: Single function or small module
- **Speed**: Fast (< 1 second each)
- **Examples**: 
  - Grid creation with correct spacing
  - Dispersion operator calculation
  - Raman response function shape

### Integration Tests
- **Purpose**: Test that components work together correctly
- **Scope**: Multiple modules, solver functionality
- **Speed**: Medium (1-10 seconds)
- **Examples**:
  - SSFM propagation without crashes
  - Energy conservation across solvers
  - Solver convergence

### Regression Tests
- **Purpose**: Ensure results match published/validated data
- **Scope**: Full simulation scenarios
- **Speed**: Slower (10-60 seconds)
- **Examples**:
  - Dudley RMP 2006 Fig. 3 supercontinuum
  - 3rd order soliton fission
  - Dispersive wave generation

## Running Tests

```julia
# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific test suite
julia --project=. test/unit/test_dispersion.jl

# Run with coverage
julia --project=. --code-coverage=user -e 'using Pkg; Pkg.test()'
```

## Test Coverage Goals

- Unit tests: 100% coverage of core functions
- Integration tests: All solvers, major workflows
- Regression tests: Key publications (Dudley 2006, Hult 2007)

## Adding New Tests

When adding features:
1. Write unit tests for new functions
2. Add integration tests if it affects solvers
3. Consider adding regression tests for validation

Follow the existing patterns in each directory.
