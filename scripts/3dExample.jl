# WARNING: this example has to be updated to fit the new BO algorithm. see 2d example.
using LinearAlgebra, Random, Distributions, Plots, GaussianProcesses
using BOOP
Random.seed!(123)

# 3D Gaussian mixture PDF
function gaussian_mixture_pdf_3d(x::Vector{Float64})
    μ1 = [-2.0, -2.0, -2.0]
    Σ1 = Matrix{Float64}(I, 3, 3) * 1.0
    μ2 = [2.0, 2.0, 2.0]
    Σ2 = Matrix{Float64}(I, 3, 3) * 1.5
    p1 = MvNormal(μ1, Σ1)
    p2 = MvNormal(μ2, Σ2)
    w1, w2 = 0.5, 0.5
    return 20 * (w1 * pdf(p1, x) + w2 * pdf(p2, x))
end

f(x) = gaussian_mixture_pdf_3d(x) + 0.0005randn()

# Setup
d = 3
lo = ones(3)*-5.0
hi = ones(3)*5.0
bounds = (lo, hi)

# Initial design
XWarm = hcat([rand(Uniform(lo[i], hi[i]), 8) for i in 1:d]...)  # 8 initial points
yWarm = [f(vec(XWarm[i, :])) for i in 1:size(XWarm, 1)]

# GP model settings
mean1 = MeanConst(0.0)
kernel1 = Mat52Ard(1. * ones(d), 1.0)
logNoise1 = log(1e-1)
KB = [[-2.0, -2.0, -2.0, -2.0], [2.5, 2.5, 2.5, 3.0]]
NB = [-4.0, 1.0]

modelSettings = (
    kernelBounds=KB,
    noiseBounds=NB,
    xBounds=[lo, hi]
)

optSettings = OptimizationSettings(
    nIter = 5,           
    n_restarts = 25,
    acq_config = EIConfig(ξ=.15) 
)


# Priors:
# Prior for Continuous Variable (x1)
# Range is [-1, 1] (width 2). We want a length scale around 0.5.
# Normal(0.0, 1.0) => Median length scale exp(0) = 1.0.
prior_cont = Normal(-0.5, 0.5)

# Prior for Signal Variance
prior_sig = Normal(0.0, 1.0)

# Apply to Kernel
# Parameter order for Mat52Ard (dim=2): [log(ℓ_x1), log(ℓ_x2), log(σ_f)]
priors = [prior_cont, prior_cont, prior_cont, prior_sig]

set_priors!(kernel1, priors)

gp_template = GPE(
    XWarm',              # Input data (transponerat för GPE)
    yWarm,               # Output data
    MeanConst(mean(yWarm)),       # Målfunktionens medelvärde (för standardiserad data)
    deepcopy(kernel1),            # Kerneln vi byggde ovan (kopieras inuti BO)
    -2.0                  # Startvärde för logNoise
)





warmStart = (XWarm, yWarm)
warmStart = (XO, yO)
gpO, XO, yO = BO(f, gp_template, modelSettings, optSettings, warmStart, DiscreteKern=0)

# Optional: Plot marginal predictions (fix x₃ = 0)

fixed_z = 1.0
function fixHeat(fixed_z; z_tol=1.0)
    
   xx = range(-5, 5, length=50)
   yy = range(-5, 5, length=50)
   
   gridd = [[x, y, fixed_z] for y in yy, x in xx];
   
   # Scale the grid
   grid_scaled = rescale.(gridd, Ref(lo), Ref(hi))
   grid_scaled_mat = reduce(hcat, grid_scaled)
   
   μ, _ = predict_f(gpO, grid_scaled_mat)
   
   # Calculate distance from the fixed z-slice
   z_dists = abs.(XO[:, 3] .- fixed_z)
   
   # Keep only points within the tolerance
   is_close = z_dists .< z_tol
   XO_slice = XO[is_close, :]
   # ---------------------------------------------------

   # Plotting mean slice
   Z = reshape(μ, length(yy), length(xx))
   
   # Evaluate true function (without noise) at grid points
   true_vals = [gaussian_mixture_pdf_3d(x) for x in gridd]
   Ztrue = reshape(true_vals, length(yy), length(xx))
   
   # Plot side-by-side comparison
   p1 = heatmap(xx, yy, Z; xlabel="x₁", ylabel="x₂", title="GP Posterior mean (z = $fixed_z)", colorbar=false, c=:viridis)
   
   # Plotting the filtered slice of samples. 
   scatter!(p1, XO_slice[:,1], XO_slice[:,2], marker=:circle, label="Samples (< $z_tol dist)", color=:white)
   
   p2 = heatmap(xx, yy, Ztrue; xlabel="x₁", ylabel="x₂", title="True function (z = $fixed_z)", colorbar=false, c=:viridis)
   scatter!(p2, XO_slice[:,1], XO_slice[:,2], marker=:circle, label="Samples (< $z_tol dist)", color=:white)
   
   plot(p1, p2, layout=(1,2), size=(900,400), c=:viridis)

end

p1 = fixHeat(-2.);
p2 = fixHeat(0.);
p3 = fixHeat(2.);
plot(p1,p2,p3,layout=(3,1), size=(900,900), color=:viridis)


allSamp = hcat(XO, yO)  # Display final samples and their function values
allSampSorted = sortslices(allSamp, dims=1, by = x -> x[4])