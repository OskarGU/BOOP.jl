using GaussianProcesses
using Random, Optim, Distributions, Plots, Statistics, LinearAlgebra
using BOOP 

Random.seed!(42)

# ==============================================================================
# PROBLEM DESCRIPTION
# ==============================================================================
# z[1]: Discrete {1, 2, 3, 4}
# z[2]: Discrete {7, 8, 9, 10, 11}
# Global Maximum: z1 = 3, z2 = 9

# Define ranges
r1 = 1:10
r2 = 4:19

# True function (Synthetic)
function true_function_2D_discrete(x_vec)
    z1 = round(x_vec[1])
    z2 = round(x_vec[2])
    
    # Peak at (3, 9)
    # Using Gaussian-like bumps on the discrete grid
    val = 5.0 * exp(-(z1 - 4)^2 / 2.5 - (z2 - 7)^2 / 4.0) + 4.0 * exp(-(z1 - 8)^2 / 1.5 - (z2 - 15)^2 / 4.0)
    
    # Add some noise
    noise = 0.5 * randn()
    return val + noise
end

# Noiseless version for plotting
f_true_plot(z1, z2) = 5.0 * exp(-(round(z1) - 4)^2 / 2.5 - (round(z2) - 7)^2 / 4.0) +4.0 * exp(-(z1 - 8)^2 / 1.5 - (z2 - 15)^2 / 4.0)

# Bounds
x_lo = [minimum(r1), minimum(r2)] # [1.0, 7.0]
x_hi = [maximum(r1), maximum(r2)] # [4.0, 11.0]
d = 2

# ==============================================================================
# INITIALIZATION (Warm Start)
# ==============================================================================
N_init = 3
X_warm = zeros(N_init, d)

# Randomly sample from the valid sets
X_warm[:, 1] = rand(r1, N_init)
X_warm[:, 2] = rand(r2, N_init)

y_warm = [true_function_2D_discrete(row) for row in eachrow(X_warm)]
warmStart = (X_warm, y_warm)

println("Startpunkter:")
display(hcat(X_warm, y_warm))

# ==============================================================================
# BO SETTINGS
# ==============================================================================

# 1. Base Kernel (ARD allows different length scales for z1 and z2)
baseKernel = Mat52Ard(zeros(d), 0.0) 

# 2. GM Kernel
# We specify that dimensions 1 AND 2 are integer dims.
# We assume GMKernel constructor accepts a tuple/vector of ranges.
GMKernel = BOOP.GarridoMerchanKernel(baseKernel, [1, 2], [r1, r2])

# 3. Model Settings
modelSettings = (
    mean = MeanConst(mean(y_warm)),
    kernel = deepcopy(GMKernel),   
    logNoise = -1.0,                
    
    # Bounds: [Log(ℓ1), Log(ℓ2), Log(σf)]
    # Important: Since variables are integers (dist 1), lower bound on length scale
    # should not be too low (< 0.0) to avoid overfitting/independence.
    kernelBounds = [[-0.5, -0.5, -2.0], [3.0, 3.0, 5.0]], 
    
    # Soft lock on noise (allow some optimization)
    noiseBounds = [-3.0, 1.0],
    
    xdim = d,
    xBounds = (x_lo, x_hi) 
)

# 4. Optimization Settings
optSettings = OptimizationSettings(
    nIter = 2,          
    n_restarts = 5,
    acq_config = EIConfig(ξ=0.2) 
)

# ==============================================================================
# PRIORS
# ==============================================================================
# Priors help guide the GP towards smoothness.
# We want length scales > 1.0 (log(1.0) = 0.0).
prior_len = Normal(1.0, 0.5) # Mean exp(1.0) ≈ 2.7 (smooth over neighbors)
prior_sig = Normal(0.0, 1.0)

# Apply priors: [ℓ1, ℓ2, σf]
set_priors!(GMKernel, [prior_len, prior_len, prior_sig])

gp_template = GPE(
    X_warm', 
    y_warm, 
    MeanConst(0.0), 
    GMKernel, 
    -1.0
)

# ==============================================================================
# RUN OPTIMIZATION
# ==============================================================================

# NOTE: DiscreteKern = [1, 2] implies that BOTH dimensions are discrete.
println("\nKör optimering...")
warmStart=(X_final, y_final)
@time gp, X_final, y_final, max_x, max_val, max_obs_x, max_obs_val = BO(
    true_function_2D_discrete, gp_template, modelSettings, optSettings, warmStart; 
    DiscreteKern=[1, 2] 
)

println("\nGlobal Model Max hittat vid: $max_x")
println("Värde vid max: $(round(max_val, digits=3))")

# ==============================================================================
# PLOTTING (Heatmap & Steps)
# ==============================================================================

# --- 1. Heatmap (True vs Model) ---
# Create a grid for plotting
grid_z1 = r1
grid_z2 = r2

Z_true = [f_true_plot(z1, z2) for z2 in grid_z2, z1 in grid_z1]
Z_model = zeros(length(grid_z2), length(grid_z1))

# Helper to predict (rescaling handling)
μ_y, σ_y = mean(y_final), std(y_final)

for (i, z2) in enumerate(grid_z2)
    for (j, z1) in enumerate(grid_z1)
        # Scale input to [-1, 1] for prediction (using your rescale logic)
        # Note: We simulate the scaling manually here to match BO internal logic
        # if using the hybrid rescale where integers are kept as integers but passed to GP.
        # Assuming GP handles the raw integers due to GM Kernel:
        pt = [Float64(z1), Float64(z2)]
        
        # If your GP was trained on [-1, 1] scaled integers, we need to scale pt here.
        # Assuming the 'hybrid' rescale where discrete are NOT scaled:
        X_in = reshape(pt, d, 1) 
        
        # But wait, your code usually scales *continuous* vars. 
        # Since DiscreteKern=[1,2], rescale() likely leaves them alone (if using the latest fix).
        # Let's assume input to predict_f should be raw integers (as floats).
        
        μ_sc, _ = predict_f(gp, X_in)
        Z_model[i, j] = μ_sc[1] * σ_y + μ_y
    end
end

p1 = heatmap(grid_z1, grid_z2, Z_true, title="Sanning", xlabel="z1 (1-4)", ylabel="z2 (7-11)", c=:viridis)
p2 = heatmap(grid_z1, grid_z2, Z_model, title="GP Modell", xlabel="z1 (1-4)", ylabel="z2 (7-11)", c=:viridis)
scatter!(p2, X_final[:,1], X_final[:,2], label="Evalueringar", mc=:red, ms=6)

# --- 2. Continuous "Staircase" Plot ---
# We slice through the space to see the GM behavior.
# Fix z2 = 9 (Optimal row), vary z1 continuously from 0.5 to 4.5
z1_cont = range(2.5, 9.5, length=300)
z2_fixed = 7.0

y_step_pred = zeros(length(z1_cont))
y_step_std = zeros(length(z1_cont))

for (i, z1) in enumerate(z1_cont)
    pt = reshape([z1, z2_fixed], 2, 1)
    μ, σ2 = predict_f(gp, pt)
    y_step_pred[i] = μ[1] * σ_y + μ_y
    y_step_std[i] = sqrt(max(σ2[1], 0)) * σ_y
end



p3 = plot(z1_cont, y_step_pred, 
    title="Snitt vid z2=7 (Visar Trappsteg)", 
    xlabel="z1 (Kontinuerlig vy)", ylabel="Output",
    lw=2, color=:blue, label="GP Mean",
    ribbon=(2*y_step_std, 2*y_step_std), fillalpha=0.2
)

# Plot True Steps
y_true_steps = [f_true_plot(z, z2_fixed) for z in z1_cont]
plot!(p3, z1_cont, y_true_steps, ls=:dash, color=:black, label="Sanning")

# Show sampled points on this slice
mask = (X_final[:, 2] .== z2_fixed)
scatter!(p3, X_final[mask, 1], y_final[mask], color=:red, ms=6, label="Data (z2=9)")

plot(p1, p2, p3, layout=@layout([a b; c]), size=(1000, 800))


######################
#########################
########################
# 1. Standard Kernel (No GM wrapper)
std_kernel = Mat52Ard(zeros(d), 0.0) 

# 2. Priors (Same as GM for fair comparison)
prior_len = Normal(1.0, 0.5) 
prior_sig = Normal(0.0, 1.0)
set_priors!(std_kernel, [prior_len, prior_len, prior_sig])

# 3. Template
gp_template_std = GPE(
    X_warm', y_warm, MeanConst(0.0), std_kernel, -1.0
)

# 4. Model Settings
modelSettings_std = (
    mean = MeanConst(mean(y_warm)),
    kernel = deepcopy(std_kernel),   
    logNoise = -1.0,                
    kernelBounds = [[-0.5, -0.5, -2.0], [3.0, 3.0, 5.0]], 
    noiseBounds = [-3.0, 1.0],
    xdim = d,
    xBounds = (x_lo, x_hi) 
)

# 5. Optimization Settings
optSettings = OptimizationSettings(
    nIter = 2,          
    n_restarts = 5,
    acq_config = EIConfig(ξ=0.2) 
)

# ==============================================================================
# RUN OPTIMIZATION (NAIVE)
# ==============================================================================

# IMPORTANT: DiscreteKern = Int[] (Empty).
# This treats all variables as continuous [-1, 1] internally.
println("Running Standard (Naive) Optimization...")
warmStart=(X_warm, y_warm)
warmStart=(X_final, y_final)
@time gp_std, X_final, y_final, max_x, max_val, _, _ = BO(
    true_function_2D_discrete, gp_template_std, modelSettings_std, optSettings, warmStart; 
    DiscreteKern=Int[] 
)

println("Max found at: $max_x")

# ==============================================================================
# PLOTTING
# ==============================================================================

μ_y, σ_y = mean(y_final), std(y_final)

# --- Helper for Manual Scaling ---
# Since DiscreteKern=[] used, the GP expects inputs in [-1, 1].
# We must scale our plotting grid manually.
function to_scaled(x_val, dim_idx)
    lo, hi = x_lo[dim_idx], x_hi[dim_idx]
    return 2 * (x_val - lo) / (hi - lo) - 1
end

# --- 1. Heatmap ---
grid_z1 = 1:6
grid_z2 = 4:11

Z_true = [f_true_plot(z1, z2) for z2 in grid_z2, z1 in grid_z1]
Z_model = zeros(length(grid_z2), length(grid_z1))

for (i, z2) in enumerate(grid_z2)
    for (j, z1) in enumerate(grid_z1)
        # Scale to [-1, 1]
        pt = reshape([to_scaled(z1, 1), to_scaled(z2, 2)], d, 1)
        μ_sc, _ = predict_f(gp_std, pt)
        Z_model[i, j] = μ_sc[1] * σ_y + μ_y
    end
end

p1 = heatmap(grid_z1, grid_z2, Z_true, title="Truth", xlabel="z1", ylabel="z2", c=:viridis)
p2=heatmap(gp_std, title="Standard GP Model", xlabel="z1", ylabel="z2", c=:viridis)
x1_scaled = 2 .* (X_final[:, 1] .- x_lo[1]) ./ (x_hi[1] - x_lo[1]) .- 1
x2_scaled = 2 .* (X_final[:, 2] .- x_lo[2]) ./ (x_hi[2] - x_lo[2]) .- 1
scatter!(p2, x1_scaled, x2_scaled, label="Evaluations", mc=:red, ms=6)

#p2 = heatmap(grid_z1, grid_z2, Z_model, title="Standard GP Model", xlabel="z1", ylabel="z2", c=:viridis)
#scatter!(p2, X_final[:,1], X_final[:,2], label="Evaluations", mc=:red, ms=6)

# --- 2. Slice Plot (Continuous view) ---
# Fix z2 = 9, vary z1 continuously to show lack of "staircase"
z1_cont = range(0.5, 4.5, length=300)
z2_fixed = 9.0

y_pred = zeros(length(z1_cont))
y_std = zeros(length(z1_cont))

for (i, z1) in enumerate(z1_cont)
    pt = reshape([to_scaled(z1, 1), to_scaled(z2_fixed, 2)], d, 1)
    μ, σ2 = predict_f(gp_std, pt)
    y_pred[i] = μ[1] * σ_y + μ_y
    y_std[i] = sqrt(max(σ2[1], 0)) * σ_y
end

p3 = plot(z1_cont, y_pred, 
    title="Slice at z2=9 (Standard GP)", 
    xlabel="z1 (Continuous View)", ylabel="Output",
    lw=2, color=:red, label="GP Mean (Smooth)",
    ribbon=(2*y_std, 2*y_std), fillalpha=0.2, fillcolor=:red
)

# Plot Truth
y_true_steps = [f_true_plot(z, z2_fixed) for z in z1_cont]
plot!(p3, z1_cont, y_true_steps, ls=:dash, color=:black, label="Truth")

# Show sampled points
mask = (round.(X_final[:, 2]) .== z2_fixed)
scatter!(p3, X_final[mask, 1], y_final[mask], color=:red, ms=6, label="Data")

plot(p1, p2, p3, layout=@layout([a b; c]), size=(1000, 800))