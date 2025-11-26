# Contributing to JuGNLSE.jl

Thank you for considering contributing to JuGNLSE.jl! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/JuGNLSE.jl.git
   cd JuGNLSE.jl
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. **Install Julia** (version 1.6 or higher)
2. **Activate the package environment**:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```
3. **Run tests** to ensure everything works:
   ```julia
   Pkg.test()
   ```

## Code Style Guidelines

### General Principles
- Follow the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- Write clear, readable code with meaningful variable names
- Keep functions focused and modular
- Use type annotations for function arguments when appropriate

### Specific Conventions
- **Indentation**: 4 spaces (no tabs)
- **Line length**: Aim for 92 characters maximum
- **Naming**:
  - Functions: `lowercase_with_underscores`
  - Types: `CamelCase`
  - Constants: `UPPERCASE_WITH_UNDERSCORES`
  - Private functions: prefix with `_`

### Example
```julia
"""
    function_name(arg1::Type1, arg2::Type2; kwarg::Type3=default)

Brief description of what the function does.

# Arguments
- `arg1::Type1`: Description of arg1
- `arg2::Type2`: Description of arg2
- `kwarg::Type3`: Description of keyword argument (default: `default`)

# Returns
- `ReturnType`: Description of return value

# Example
\```julia
result = function_name(val1, val2)
\```
"""
function function_name(arg1::Type1, arg2::Type2; kwarg::Type3=default)
    # Implementation
    return result
end
```

## Documentation

- **All public functions** must have docstrings
- Use the format shown above with sections:
  - Brief description
  - `Arguments` section
  - `Returns` section
  - `Example` section (optional but encouraged)
- Update documentation in `docs/src/` if adding new features

## Testing

### Writing Tests
- Add tests for all new functionality in `test/`
- Use descriptive test names
- Group related tests in `@testset` blocks
- Aim for high code coverage

### Running Tests
```julia
using Pkg
Pkg.test("JuGNLSE")
```

### Example Test
```julia
@testset "New Feature" begin
    # Setup
    grid = create_grid(2^10, 10e-12, 835e-9)
    
    # Test
    @test grid.N == 1024
    @test length(grid.t) == 1024
    
    # Test error handling
    @test_throws ArgumentError create_grid(-1, 10e-12, 835e-9)
end
```

## Performance Considerations

- **Profile before optimizing**: Use `@profile` and `@benchmark`
- **Type stability**: Avoid type instabilities (check with `@code_warntype`)
- **Pre-allocate arrays**: Use in-place operations where possible
- **Use views**: Prefer `@views` for array slicing
- **SIMD**: Use `@.` macro for vectorized operations

### Benchmarking Example
```julia
using BenchmarkTools

@benchmark your_function(args)
```

## Pull Request Process

1. **Update tests**: Ensure all tests pass
2. **Update documentation**: Add/update docstrings and docs
3. **Update CHANGELOG**: Add entry describing your changes
4. **Commit messages**: Write clear, descriptive commit messages
   ```
   Add feature X for Y
   
   - Detailed point 1
   - Detailed point 2
   
   Closes #123
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** on GitHub
7. **Respond to review feedback**

## Types of Contributions

### Bug Reports
- Use the GitHub issue tracker
- Include:
  - Julia version
  - JuGNLSE version
  - Minimal reproducible example
  - Error message/stack trace

### Feature Requests
- Open an issue first to discuss
- Explain the use case and benefits
- Be open to feedback

### Code Contributions
Priority areas:
- Additional Raman models
- Coupled-mode GNLSE
- GPU acceleration
- Visualization utilities
- Performance optimizations
- Documentation improvements

## Code Review

All contributions will be reviewed for:
- **Correctness**: Does it work as intended?
- **Performance**: Is it efficient?
- **Style**: Does it follow conventions?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it well-documented?

## Questions?

- Open an issue on GitHub
- Check existing documentation
- Review similar implementations in `examples/`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Contributors will be acknowledged in:
- README.md
- Documentation
- Release notes

Thank you for contributing to JuGNLSE.jl!
