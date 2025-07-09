Random.seed!(123)
using LinearAlgebra
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

f(x) = -gaussian_mixture_pdf_3d(x) + 0.05randn()

# Setup
d = 3
lo = -5.0
hi = 5.0
bounds = (lo, hi)

# Initial design
X = rand(Uniform(lo, hi), 8, d)
y = [f(vec(X[i, :])) for i in 1:size(X, 1)]

# GP model settings
mean1 = MeanConst(0.0)
kernel1 = Mat52Ard(3 * ones(d), 1.0)
logNoise1 = log(1e-1)
KB = [[-3.0, -3.0, -3.0, -8.0], [3.0, 3.0, 3.0, 8.0]]
NB = [-6.0, 3.0]

modelSettings = (
    mean=mean1,
    kernel=kernel1,
    logNoise=logNoise1,
    kernelBounds=KB,
    noiseBounds=NB,
    xdim=d,
    xBounds=[lo, hi]
)

optimizationSettings = (
    nIter=50,
    tuningPar=0.05,
    n_restarts=20,
    bounds=(-1.0, 1.0),
    acq=expected_improvement
)

warmStart = (X, y)
warmStart = (XO, yO)
gpO, XO, yO = BO(f, modelSettings, optimizationSettings, warmStart)

# Optional: Plot marginal predictions (fix x₃ = 0)
xx = range(lo, hi, length=50)
yy = range(lo, hi, length=50)
fixed_z = -2.0
grid = [[x, y, fixed_z] for y in yy, x in xx]
grid_scaled = rescale.(grid, lo, hi)
grid_scaled_mat = reduce(hcat, grid_scaled)
μ, _ = predict_f(gpO, grid_scaled_mat)

# Plotting mean slice
Z = reshape(μ, length(yy), length(xx))
heatmap(xx, yy, Z; xlabel="x₁", ylabel="x₂", title="Posterior mean (x₃ = 0)")
scatter!(XO[:,1], XO[:,2], marker=:circle, label="Samples", color=:white)


# Evaluate true function (without noise) at grid points
true_vals = [-gaussian_mixture_pdf_3d(x) for x in grid]
Ztrue = reshape(true_vals, length(yy), length(xx))

# Plot side-by-side comparison
p1 = heatmap(xx, yy, Z'; xlabel="x₁", ylabel="x₂", title="GP Posterior mean (x₃ = 0)", colorbar=false)
scatter!(p1, XO[:,2], XO[:,1], marker=:circle, label="Samples", color=:white)

p2 = heatmap(xx, yy, Ztrue; xlabel="x₁", ylabel="x₂", title="True function (x₃ = 0)", colorbar=false)
scatter!(p2, XO[:,2], XO[:,1], marker=:circle, label="Samples", color=:white)

plot(p1, p2, layout=(1,2), size=(900,400))

allSamp = hcat(XO, yO)  # Display final samples and their function values
allSampSorted = sortslices(allSamp, dims=1, by = x -> x[4])