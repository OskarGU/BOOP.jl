##################################################
using Random
using Distributions
using Plots
using GaussianProcesses
using BOOP
#using UniformScaling # Needed for some covariance matrix definitions

# Assuming your module is loaded, e.g.:
# include("src/MyBayesianOpt.jl")
# using .MyBayesianOpt

# --- 1. Define the Problem ---
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
    # Scaled by 20 to make the peaks more pronounced
    return 20 * (w1 * pdf(p1, x) + w2 * pdf(p2, x))
end

# Noisy version of the function
f(x) = gaussian_mixture_pdf(x) + 0.05 * randn()

# Define problem dimensions and bounds
d = 2
lower = [-5.0, -6.0]
upper = [6.0, 7.0]
bounds = (lower, upper)

# Initial design: 5 random points
X_warm = hcat([rand(Uniform(l, u), 5) for (l, u) in zip(bounds[1], bounds[2])]...)
y_warm = [f(vec(X_warm[i, :])) for i in 1:size(X_warm, 1)]
warmStart = (X_warm, y_warm)


# --- 2. Define the GP Model Settings ---
# This part remains the same
modelSettings = (
    mean = MeanConst(0.0),
    kernel = SEArd(log(1.0) * ones(d), 0.0), # SE kernel with ARD
    logNoise = -1.0,
    # Note: SEArd has d+1 params (d lengthscales, 1 signal variance)
    kernelBounds = [[-3., -3., -5.], [log(1.5), log(1.5), 2.]],
    noiseBounds = [-6., 0.1],
    xdim = d,
    xBounds = bounds
)


# --- 3. Define the Optimization Settings (NEW STRUCTURE) ---

# Example A: Expected Improvement
opt_settings_ei = OptimizationSettings(
    nIter = 3,
    n_restarts = 30,
    acq_config = EIConfig(ξ=0.3)
)

# Example B: Upper Confidence Bound
opt_settings_ucb = OptimizationSettings(
    nIter = 3,
    n_restarts = 20,
    acq_config = UCBConfig(κ=3.5)
)

# Example C: Knowledge Gradient Discrete
# Create a 2D grid in the scaled [-1, 1] space
x_coords = range(-1.0, 1.0, length=50)
y_coords = range(-1.0, 1.0, length=50)
grid_iterator = Iterators.product(x_coords, y_coords)
grid_2d_scaled = hcat([[x, y] for (x, y) in grid_iterator]...)

opt_settings_kgd = OptimizationSettings(
    nIter = 3, # KGD is slow, so fewer iterations
    n_restarts = 10,
    acq_config = KGDConfig(domain_points=grid_2d_scaled)
)

opt_settings_kgh = OptimizationSettings(
    nIter = 3,    
    n_restarts = 10,  
    acq_config = KGHConfig(
        n_z = 40    )
)

opt_settings_kgq = OptimizationSettings(
    nIter = 2,       # Antal BO-iterationer
    n_restarts = 10,  # Antal starter för att optimera acquisition-funktionen
    acq_config = KGQConfig(
        n_z = 20,      # Fler punkter för en bättre integral-approximation
        alpha = 0.8,   # Starkt fokus på svansarna (mer exploration)
        n_starts = 10  # Antal starter för *varje* inre max-problem
    )
)

# Välj denna konfiguration för din körning
chosen_settings = opt_settings_kgq

# --- 4. Run Bayesian Optimization ---
# Choose ONE of the settings from above to run
chosen_settings = opt_settings_ei

println("Running Bayesian Optimization with $(typeof(chosen_settings.acq_config))...")
warmStart = (X_warm, y_warm)

warmStart = (X_final, y_final)
# Note: The BO function returns maximizers, not minimizers
@time gp_final, X_final, y_final, maximizer_global, global_max, maximizer_observed, observed_max = BO(
    f, modelSettings, chosen_settings, warmStart
);


# --- 5. Plot the Results ---

# Create grid for plotting
xx = range(bounds[1][1], bounds[2][1], length=100);
yy = range(bounds[1][2], bounds[2][2], length=100);

# Compute true function on grid
Z_true = [gaussian_mixture_pdf([x, y]) for y in yy, x in xx];

# Predict GP mean on grid
grid_points = [[x, y] for x in xx, y in yy];
X_grid = reduce(hcat, grid_points);
μ, _ = predict_f(gp_final, rescale(X_grid', bounds[1], bounds[2])');

# Un-standardize predictions
μ_y, σ_y = mean(y_final), std(y_final);
μ_unscaled = μ .* σ_y .+ μ_y;
Z_gp = reshape(μ_unscaled, 100, 100);

# Plot true function heatmap (we plot the positive function)
p1 = heatmap(xx, yy, Z_true', title="True Function PDF", xlabel="x₁", ylabel="x₂", c=:viridis)

# Plot GP mean heatmap with samples
p2 = heatmap(xx, yy, Z_gp', title="GP Mean Posterior", xlabel="x₁", ylabel="x₂", c=:viridis)
scatter!(p2, X_final[:, 1], X_final[:, 2], color=:white, markersize=3, label="Samples", msw=0.5, msc=:black)
scatter!(p2, [X_final[end, 1]], [X_final[end, 2]], color=:red, markersize=5, label="Last Sample", msw=1.0, msc=:black)

plot(p1, p2, layout=(1, 2), size=(1200, 500))

