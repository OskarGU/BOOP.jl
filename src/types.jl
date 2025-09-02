# 1. Abstract parent type
abstract type AcquisitionConfig end

# Create a new abstract type just for KG methods
abstract type KnowledgeGradientConfig <: AcquisitionConfig end


# 2. Concrete structs for each acquisition function
@kwdef struct EIConfig <: AcquisitionConfig
    ξ::Float64
end
# Add a convenient constructor with a default value
EIConfig(; ξ::Float64 = 0.01) = EIConfig(ξ)

@kwdef struct UCBConfig <: AcquisitionConfig
    κ::Float64
end
UCBConfig(; κ::Float64 = 2.0) = UCBConfig(κ)

# Make the concrete KG structs subtypes of the new abstract type
@kwdef struct KGHConfig <: KnowledgeGradientConfig
    n_z::Int
end
KGHConfig(; n_z::Int = 5) = KGHConfig(n_z)

@kwdef struct KGDConfig <: KnowledgeGradientConfig
    domain_points::Matrix{Float64}
end

@kwdef struct PosteriorVarianceConfig <: AcquisitionConfig end

@kwdef struct OptimizationSettings
    nIter::Int
    n_restarts::Int
    acq_config::AcquisitionConfig # This will hold one of the structs from above
end