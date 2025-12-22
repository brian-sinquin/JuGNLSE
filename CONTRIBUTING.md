# Contributing to JuGNLSE.jl

Contributions to JuGNLSE.jl are welcome. Please follow these simple guidelines to ensure a smooth process.

## Development Workflow

1. **Fork and Clone**: Fork the repository and clone it to your local machine.
2. **Environment**: Activate the project environment in Julia:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```
3. **Branching**: Create a new branch for your changes.
4. **Testing**: Ensure all tests pass before submitting a pull request:
   ```julia
   Pkg.test()
   ```

## Guidelines

- **Code Style**: Follow standard Julia conventions. Aim for clarity and type stability.
- **Documentation**: Provide docstrings for any new public functions.
- **Tests**: Include tests for new features or bug fixes in the `test/` directory.
- **Pull Requests**: Keep pull requests focused on a single change. Provide a clear description of the modifications.

Thank you for helping improve JuGNLSE.jl.
