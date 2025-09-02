

using GaussianProcesses
using Random, Optim, Distributions, Plots
using BOOP

# Set random seed for reproducibility
Random.seed!(123)

# Define black-box objective function (you can change this!)
f(x) = -(1.5*sin(3x) + 0.5x^2 - x + 0.2randn())
fNf(x) = -(1.5*sin(3x) + 0.5x^2 - x )


# Bounds
lo, hi = -4.0, 5.0

# Initial design points (column vector form)
X = reshape([-2.5, 0.0, 4.0], :, 1)
y = f.(X)
Xscaled= rescale(X, lo, hi)

# Define GP components
mean1 = MeanConst(0.0)
kernel1 = Mat52Ard([0.0], 1.0)
logNoise1 = log(0.1)


d=1
mean1 = MeanConst(0.0)
kernel1 = Mat52Ard(3*ones(d), 1.0)  # lengthscale zeros means will be optimized
logNoise1 = log(1e-1)              # low noise since pdf is deterministic
KB = [[-3, -5], [3, 3]];
NB = [-6., 3.];

lo=-4.; hi=5.;

# Number of optimization iterations
nIter = 1

modelSettings = (mean=mean1, kernel = kernel1, logNoise = logNoise1, 
                 kernelBounds = KB, noiseBounds=NB, xdim=d, xBounds=[lo, hi]

)

# Optimization settings
optimizationSettings = (nIter=1, tuningPar=0.02,  n_restarts=20, acq=expected_improvement, nSim=nothing, nq=nothing, dmp=nothing)
optimizationSettings = (nIter=1, tuningPar=2.0,  n_restarts=20, acq=upper_confidence_bound, nSim=nothing, nq=nothing, dmp=nothing)
optimizationSettings = (nIter=1, tuningPar=nothing,  n_restarts=10, acq=knowledgeGradientHybrid, nq=80, dmp=nothing)
optimizationSettings = (nIter=1, tuningPar=nothing,  n_restarts=10, acq=knowledgeGradientDiscrete, 
   nSim=nothing, nq=nothing, dmp=Matrix(reshape(range(-1,1,length=1000), 1, :)))


warmStart = (X, y[:])
warmStart = (XO, yO)

@time gpO, XO, yO, objMin, obsMin, postMaxObsY  = BO(f, modelSettings, optimizationSettings, warmStart);


##################
# Probably make plots to one function and put it in a "plotUtils" script.
# Create a dense grid in the original input space
x_plot = range(lo, hi, length=200);
x_plot_array = collect(x_plot);  # needed for broadcasting

# Rescale to [-1, 1]
x_plot_scaled = rescale(x_plot_array, lo, hi);
x_plot_scaled_matrix = reshape(x_plot_scaled, 1, :);

# Predict from GP
μScaled, σ²Scaled = predict_f(gpO, x_plot_scaled_matrix);
μ_y = mean(yO)
σ_y = max(std(yO), 1e-6)

# Transform predictions back to the original scale
μ = vec(μScaled) .* σ_y .+ μ_y
σ² = vec(σ²Scaled) .* (σ_y^2)

# Plot in original coordinates
plot(x_plot, μ; ribbon=sqrt.(σ²), label="GP prediction", lw=2, xlabel="x");
plot!(x_plot, fNf.(x_plot), label="True function f(x)", lw=2);
xlabel!("x")
ylabel!("f(x)")
title!("GP vs True Function")
scatter!(XO, yO, label="Samples", color=:black)
vline!([obsMin], label="Observed Maximizer", color=:red, linestyle=:dash)
vline!([objMin], label="Objective Maximizer", color=:blue, linestyle=:dash)
hline!([postMaxObsY], label=" Posterior Maximum", color=:green, linestyle=:dash)
###################



###########
##########
#############
# Set random seed for reproducibility
Random.seed!(123)

# Define black-box objective function
# Note: The BO function expects a function that takes a vector `x` as input.
f(x::AbstractVector) = -(1.5*sin(3*x[1]) + 0.5*x[1]^2 - x[1] + 0.2*randn())
f_true(x) = -(1.5*sin(3*x) + 0.5*x^2 - x)

# Define the search domain bounds
lo, hi = -4.0, 5.0
d = 1 # Dimensionality

# Create initial data (warm start)
X_warm = reshape([-2.5, 0.0, 4.0], :, 1)
y_warm = vec([f(x) for x in eachrow(X_warm)]) # Ensure y is a vector
warmStart = (X_warm, y_warm)


# --- 2. Define the GP Model Settings ---
modelSettings = (
    mean = MeanConst(0.0),
    kernel = Mat52Ard(zeros(d), 0.0), # Initialize ls and σ_f to be optimized
    logNoise = -1.0,                 # Initial guess for log noise
    kernelBounds = [[-3.0, -5.0], [3.0, 3.0]],  # Bounds for log lengthscale
    noiseBounds = [-4.0, 2.0],       # Bounds for log noise
    xdim = d,
    xBounds = ([lo], [hi])           # Use tuple of vectors for consistency
)


# --- 3. Define the Optimization Settings (NEW STRUCTURE) ---

# Example A: Expected Improvement (EI)
opt_settings_ei = OptimizationSettings(
    nIter = 1,
    n_restarts = 20,
    acq_config = EIConfig(ξ=0.01) # Contains only EI-specific parameters
)

# Example B: Upper Confidence Bound (UCB)
opt_settings_ucb = OptimizationSettings(
    nIter = 1,
    n_restarts = 20,
    acq_config = UCBConfig(κ=2.5) # Contains only UCB-specific parameters
)

# Example C: Knowledge Gradient Hybrid (KGH)
opt_settings_kgh = OptimizationSettings(
    nIter = 1,
    n_restarts = 10,
    acq_config = KGHConfig(n_z=30) # Contains only KGH-specific parameters
)

# Example D: Knowledge Gradient Discrete (KGD)
# Define the discrete points in the SCALED [-1, 1] domain
domain_points = reshape(range(-1, 1, length=1500), 1, :)
opt_settings_kgd = OptimizationSettings(
    nIter = 1,
    n_restarts = 10,
    acq_config = KGDConfig(domain_points=domain_points) # Contains only KGD-specific parameters
)


# --- 4. Run Bayesian Optimization ---
# Choose ONE of the settings from above to run the optimization
chosen_settings = opt_settings_kgh

println("Running Bayesian Optimization with $(typeof(chosen_settings.acq_config))...")

warmStart = (X_final, y_final)
@time gp, X_final, y_final, maximizer_global, maximizer_observed, value_observed = BO(
    f, modelSettings, chosen_settings, warmStart
);


# --- 6. Plot the Results (Optional) ---
x_grid = range(lo, hi, length=200);
X_grid_scaled = rescale(x_grid', [lo], [hi]);

# Get posterior predictions from the final GP model
μ_final, σ²_final = predict_f(gp, X_grid_scaled);
σ_final = sqrt.(max.(σ²_final, 0.0));

# Un-standardize the predictions to match the original y-scale
μ_y_final, σ_y_final = mean(y_final), std(y_final);
μ_unscaled = μ_final .* σ_y_final .+ μ_y_final;
σ_unscaled = σ_final .* σ_y_final;

plot(x_grid, f_true, label="True Function", lw=2, color=:black, legend=:bottom)
plot!(x_grid, μ_unscaled, label="Posterior Mean", ribbon=(2*σ_unscaled, 2*σ_unscaled), lw=2, c=1)
scatter!(X_warm, y_warm, label="Initial Points", markersize=5, color=:red)
scatter!(X_final, y_final, label="Observed Points", markersize=5, color=:green)
vline!([maximizer_observed[1]], label="Best Observed Point", lw=2, ls=:dash, color=:purple)
title!("Bayesian Optimization Results")
xlabel!("x")
ylabel!("f(x)")