# One dimensional example of Bayesian Optimization using BOOP.jl.
using GaussianProcesses
using Random, Optim, Distributions, Plots
using BOOP

# Set random seed for reproducibility
Random.seed!(123)

# Defineobjective function
f(x) = -(1.5*sin(3*x[1]) + 0.5*x[1]^2 - x[1] + 0.2*randn())
f_true(x) = -(1.5*sin(3*x) + 0.5*x^2 - x)


# --- 1. Set up the problem ---
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


# --- 3. Define the Optimization Settings ---

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
    acq_config = KGHConfig(n_z=10) # Contains only KGH-specific parameters
)

# Example D: Knowledge Gradient Discrete (KGD)
# Define the discrete points in the SCALED [-1, 1] domain
domain_points = reshape(range(-1, 1, length=1500), 1, :)
opt_settings_kgd = OptimizationSettings(
    nIter = 1,
    n_restarts = 10,
    acq_config = KGDConfig(domain_points=domain_points) # Contains only KGD-specific parameters
)

# Example E: Knowledge Gradient Quadrature (KGQ)
opt_settings_kgq = OptimizationSettings(
    nIter = 1,       # Antal BO-iterationer
    n_restarts = 10,  # Antal starter för att optimera acquisition-funktionen
    acq_config = KGQConfig(
        n_z = 25,      # Fler punkter för en bättre integral-approximation
        alpha = 0.8,   # Starkt fokus på svansarna (mer exploration)
        n_starts = 10  # Antal starter för *varje* inre max-problem
    )
)


# --- 4. Run Bayesian Optimization ---
# Choose ONE of the settings from above to run the optimization
chosen_settings = opt_settings_kgq

println("Running Bayesian Optimization with $(typeof(chosen_settings.acq_config))...")
warmStart = (X_warm, y_warm)
warmStart = (X_final, y_final)
@time gp, X_final, y_final, maximizer_global, global_max, maximizer_observed, observed_max = BO(
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