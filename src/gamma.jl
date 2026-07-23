
"""
    gamma(gamma_coeff::ConstantGamma, lambda::Real, z::Real)

Returns the constant nonlinear coefficient from a `ConstantGamma` model.
"""
function gamma(gamma_coeff::ConstantGamma, lambda::Real, z::Real)
    return gamma_coeff.gamma
end

"""
    gamma(gamma_coeff::ZDependentGamma, lambda::Real, z::Real)

Returns the nonlinear coefficient from a `ZDependentGamma` model at a given `z`.
"""
function gamma(gamma_coeff::ZDependentGamma, lambda::Real, z::Real)
    return gamma_coeff.gamma_func(z)
end

"""
    gamma(gamma_coeff::WavelengthDependentGamma, lambda::Real, z::Real)

Returns the nonlinear coefficient from a `WavelengthDependentGamma` model at a given `lambda`.
"""
function gamma(gamma_coeff::WavelengthDependentGamma, lambda::Real, z::Real)
    return gamma_coeff.gamma_func(lambda)
end
