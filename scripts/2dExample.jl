#########################
# 2d case
# Set random seed for reproducibility
Random.seed!(123)

# Bivariate Gaussian mixture pdf (objective function)
function gaussian_mixture_pdf(x::Vector{Float64})
    μ1 = [-2.0, -2.0]
    Σ1 = [2.0 1.5; 1.5 2.0]
    μ2 = [2.0, 2.0]
    Σ2 = [2.0 -1.7; -1.7 2.0]
    p1 = MvNormal(μ1, Σ1)
    p2 = MvNormal(μ2, Σ2)
    w1, w2 = 0.5, 0.5
    return 15*(w1 * pdf(p1, x) + w2 * pdf(p2, x))
end

f(x) = -gaussian_mixture_pdf(x) + 0.05randn()




# Posterior minimum only on observed points
function posteriorMinObs(gp, X)
    μ, _ = predict_f(gp, X')
    return minimum(μ)
end



# Initialize parameters
d = 2
bounds = (-5.0, 5.0)

# Initial design: 5 random points in 2D within bounds
X = rand(Uniform(bounds[1], bounds[2]), 5, d)
y = [f(vec(X[i, :])) for i in 1:size(X, 1)]

# Rescale to 0,1
hi = 5.; lo=-5.

d=2
mean1 = MeanConst(0.0)
kernel1 = Mat52Ard(3*ones(d), 1.0)  # lengthscale zeros means will be optimized
logNoise1 = log(1e-1)              # low noise since pdf is deterministic
KB = [[-3, -3., -8], [3, 3., 8]];
NB = [-6., 3.];

lo=-5.; hi=5.;

modelSettings = (mean=mean1, kernel = kernel1, logNoise = logNoise1, 
                 kernelBounds = KB, noiseBounds=NB, xdim=d, xBounds=[lo, hi]

)


# The results are quite sensitive to the tuning parameter in the 2d case. 
# Especially ucb is sensitive.
optimizationSettings = (nIter=20, tuningPar=0.2,  n_restarts=20, bounds=(-1.0, 1.0), 
                        acq=expected_improvement
)

optimizationSettings = (nIter=20, tuningPar=3.5,  n_restarts=20, bounds=(-1.0, 1.0), 
                        acq=upper_confidence_bound
)

warmStart = (X, y)
warmStart = (XO, yO)
gpO, XO, yO = BO(f, modelSettings, optimizationSettings, warmStart)







# Create grid for plotting
xx = range(bounds[1], bounds[2], length=100)
yy = range(bounds[1], bounds[2], length=100)

# Compute true function on grid
Z_true = [gaussian_mixture_pdf([x,y]) for y in yy, x in xx]  # (y,x) indexing for heatmap

# Predict GP mean on grid
grid_points = [[x, y] for y in yy, x in xx]
X_grid = reduce(hcat, grid_points)  # 2 x 10000
μ, _ = predict_f(gpO, rescale.(X_grid', lo, hi)')
Z_gp = reshape(μ, 100, 100)'

# Plot true function heatmap
p1 = heatmap(xx, yy, -Z_true, title="True function (Gaussian Mixture PDF)", xlabel="x", ylabel="y")

# Plot GP mean heatmap with samples overlaid
p2 = heatmap(xx, yy, Z_gp', title="GP mean posterior", xlabel="x", ylabel="y")
scatter!(p2, XO[:,1], XO[:,2], color=:white, markersize=5, label="Samples")
scatter!(p2, [XO[end,1]], [XO[end,2]], color=:red, markersize=8, label="Last samples")

plot(p1, p2, layout=(1,2), size=(1000,400))






# Noisy heatmap, how data looks:
ZNoise = [-f([x,y]) for y in yy, x in xx]  # (y,x) indexing for heatmap
p1 = heatmap(xx, yy, ZNoise, title="True function (Gaussian Mixture PDF)", xlabel="x", ylabel="y")





