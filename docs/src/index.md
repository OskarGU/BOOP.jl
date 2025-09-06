```@meta
CurrentModule = BOOP
```

# BOOP

Documentation for [BOOP](https://github.com/OskarGU/BOOP.jl).

This is a get starting message to see that every thing works and we can add stuff.


# Example usage:

## Bayesian Optimization using *BO()*

```julia
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

chosen_settings = opt_settings_kgd

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
scatter!(X_final, y_final, label="Observed Points", markersize=5, color=:green,alpha=0.5)
vline!([maximizer_observed[1]], label="Best Observed Point", lw=2, ls=:dash, color=:purple)
title!("Bayesian Optimization Results")
xlabel!("x")
ylabel!("f(x)")
```
We start by simulating a time-varying AR(2) model for illustration. We use 1000 observation and the first AR-coefficient evolves as:

$ \phi_{1,t} = sin\left(\frac{4πt}{T} \right) + 0.4, \quad \text{for }  t \in 1,\dots,1000 $





Some more info




```@index
```

