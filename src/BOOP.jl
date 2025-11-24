module BOOP

# Using other packages
using LinearAlgebra, Distributions, Random, Statistics, Optim, Plots, SpecialFunctions, StatsBase
using GaussianProcesses, ForwardDiff, LatinHypercubeSampling

# Write your package code here.

include("kernels.jl")
export GarridoMerchanKernel

include("types.jl")
export EIConfig, UCBConfig, KGHConfig, KGDConfig, KGQConfig, PosteriorVarianceConfig, OptimizationSettings

include("acquisitionfunctions.jl")
export expected_improvement, upper_confidence_bound, knowledge_gradient, knowledgeGradientMonteCarlo, 
knowledgeGradientDiscrete, knowledgeGradientHybrid, multi_start_maximize, posterior_max, ExpectedMaxGaussian,
posteriorMax, posterior_variance, knowledgeGradientQuadrature

include("bayesoptfunctions.jl")
export BO, rescale, inv_rescale, propose_next




end
