# 1. Abstract parent type
abstract type AcquisitionConfig end

# Create a new abstract type just for KG methods
abstract type KnowledgeGradientConfig <: AcquisitionConfig end


# 2. Concrete structs for each acquisition function

# Rättad version: Standardvärde direkt i structen
@kwdef struct EIConfig <: AcquisitionConfig
    ξ::Float64 = 0.01
end

# Rättad version
@kwdef struct UCBConfig <: AcquisitionConfig
    κ::Float64 = 2.0
end

# Make the concrete KG structs subtypes of the new abstract type
# Rättad version
@kwdef struct KGHConfig <: KnowledgeGradientConfig
    n_z::Int = 5
end

@kwdef struct KGDConfig <: KnowledgeGradientConfig
    domain_points::Matrix{Float64}
end

@kwdef struct PosteriorVarianceConfig <: AcquisitionConfig end

@kwdef struct OptimizationSettings
    nIter::Int
    n_restarts::Int
    acq_config::AcquisitionConfig # This will hold one of the structs from above
end

# Denna var redan korrekt och är ett bra exempel
@kwdef struct KGQConfig <: KnowledgeGradientConfig # Q för Quadrature
    n_z::Int = 30
    alpha::Float64 = 0.5
    n_starts::Int = 15
end