module BOOP

# Using other packages
using LinearAlgebra, Distributions, Random, Statistics, Optim, Plots, SpecialFunctions
using GaussianProcesses

# Write your package code here.
include("FirstFunc.jl")
export first_func

end
