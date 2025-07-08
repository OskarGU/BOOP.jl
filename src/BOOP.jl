module BOOP

# Using other packages
using LinearAlgebra, Distributions, Random, Statistics, Optim, Plots, SpecialFunctions
using GaussianProcesses

# Write your package code here.
include("FirstFunc.jl")
export first_func

include("acquisitionfunctions.jl")
export expected_improvement, upper_confidence_bound

include("bayesoptfunctions.jl")
export BO, rescale, inv_rescale, propose_next



end
