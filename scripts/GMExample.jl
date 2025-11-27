using GaussianProcesses
using Random, Optim, Distributions, Plots, Statistics, LinearAlgebra
using BOOP 


Random.seed!(41)

# Problem description
# x[1] (continuous): [-1, 1]
# x[2] (Discrete):  [1, 13] (integers)
# Global maximum: x ≈ 0.5, z = 7.

function true_function(x_vec)
    x_cont = x_vec[1]
    z_int  = round(x_vec[2]) # Force integers
    
    disc_part = exp(-(z_int - 7)^2 / 8.0)
    cont_part = sqrt(abs(x_cont))*cos(3 * (x_cont*1.3 - 0.5))
    
    # Some measurement noise
    noise = 0.25 * randn()
    
    # Scale up a bit.
    return 10.0 * disc_part * cont_part + noise
end

# No noise version for plotting
f_true_plot(x, z) = 10.0 * exp(-(z - 7)^2 / 8.0) * sqrt(abs(x))*cos(3 * (x*1.3 - 0.5)) 

# Optimization bounds.
x_lo, x_hi = -1.0, 1.0
z_lo, z_hi = 2, 12
d = 2

# Generate starting values
N_init = 5
X_warm = zeros(N_init, d)
X_warm[:, 1] = rand(Uniform(x_lo, x_hi), N_init)   
X_warm[:, 2] = rand(z_lo:z_hi, N_init)              
y_warm = [true_function(row) for row in eachrow(X_warm)]

warmStart = (X_warm, y_warm)

# ==============================================================================
# BO SETTINGS
# ==============================================================================

# Set up the kernel with Garrido-Merchán adjustments
# Use Matern52Ard ase base-kernel.
base_k = Mat52Ard([0.1;0.6], 0.8) 
GMKernel = BOOP.GarridoMerchanKernel(base_k, [2], [z_lo:z_hi])


modelSettings = (
    mean = MeanConst(mean(y_warm)),
    kernel = deepcopy(GMKernel),   
    logNoise = -2.0,                
    kernelBounds = [[-3.0, 0.5, -2.0], [3.0, 3.0, 5.0]], 
    noiseBounds = [-5.0, 2.0],
    xdim = d,
    # Bounds for rescaleing.
    xBounds = ([x_lo, x_hi]) 
)

opt_settings = OptimizationSettings(
    nIter = 3,          
    n_restarts = 10,
    acq_config = EIConfig(ξ=0.1) 
    #acq_config = UCBConfig(κ=2.5)
    #acq_config = KGHConfig(n_z=50)
)

# ==============================================================================
# Run BAYESIAN OPTIMIZATION
# ==============================================================================


# IMPORTANT: DiscreteKern=1 makes sure that the last dimension does not rescale.
warmStart = (X_final, y_final)
@time gp, X_final, y_final, max_x, max_val, max_obs_x, max_obs_val = BO(
    true_function, modelSettings, opt_settings, warmStart; DiscreteKern=1
)

println("\nGlobal Model Max hittat vid: $max_x")
println("Värde vid max: $(round(max_val, digits=3))")

# ==============================================================================
# PLOTTING RESULTS
# ==============================================================================

# Grids
xs = range(x_lo, x_hi, length=100)
zs = z_lo:z_hi

# Colect mean and std of final observations for rescaling plots
μ_y, σ_y = mean(y_final), std(y_final)


# Evaluate true function on the grid.
Z_true = [f_true_plot(x, z) for z in zs, x in xs] # Matris för sanningen

# Evaluate GP model on the grid.
Z_model = zeros(length(zs), length(xs))
for (i, z) in enumerate(zs)
    for (j, x) in enumerate(xs)
        pt = [x, Float64(z)] 
        μ_scaled, _ = predict_f(gp, reshape(pt, d, 1))
        # The GP output from BO is for scaled data, so we need to rescale back the predictions.
        Z_model[i, j] = μ_scaled[1] * σ_y + μ_y
    end
end

p1 = heatmap(xs, zs, Z_true, title="Sanning", xlabel="x (Kont)", ylabel="z (Diskret)", c=:viridis);
p2 = heatmap(xs, zs, Z_model, title="GP Modell", xlabel="x (Kont)", ylabel="z (Diskret)", c=:viridis);
# Add sampled points.
scatter!(p2, X_final[:,1], X_final[:,2], label="Sampel", mc=:red, ms=4, legend=false);
plot(p1,p2, size=(1400,400));

# Slice plots at specific discrete z-values
z_slices = [5, 7, 8] # 7 is optimum
slice_plots = []

for z_val in z_slices
    y_slice_true = [f_true_plot(x, z_val) for x in xs]
    
    pts = hcat(collect(xs), fill(float(z_val), length(xs)))'
    
    # Predict scaled values
    μ_pred_sc, σ2_pred_sc = predict_f(gp, pts)
    σ_pred_sc = sqrt.(max.(σ2_pred_sc, 0.0))
    
    # Scale back to original y-scale.
    y_slice_model = μ_pred_sc .* σ_y .+ μ_y
    σ_slice_model = σ_pred_sc .* σ_y
    
    # Check which samples correspond to this z-value
    is_at_z = (round.(X_final[:, 2]) .== z_val)
    samples_x = X_final[is_at_z, 1]
    samples_y = y_final[is_at_z]

    p = plot(xs, y_slice_true, label="Sanning (z=$z_val)", lw=2, lc=:black, title="Snitt vid z = $z_val", ylims=(-10, 8))
    plot!(p, xs, y_slice_model, label="GP Mean", lw=2, lc=:blue, ribbon=(1.96*σ_slice_model, 1.96*σ_slice_model), fillalpha=0.2, fc=:blue)
    scatter!(p, samples_x, samples_y, label="Observerat", mc=:red, ms=6)
    push!(slice_plots, p)
end;

# Combine the subplots
l = @layout [a b; c d e]
final_plot = plot(p1, p2, slice_plots..., layout=l, size=(1200, 1000));



# Vary z continuously to see the step behavior
z_plot_grid = range(0, 13, length=500)

# Select 4 different fixed x-values to plot the steps at.
x_fixed_values = [-0.6, 0.9, 0.5, 0.9]

# Layout 2x2
plot_layout = @layout [a b; c d]
p_steps = plot(layout=plot_layout, size=(1000, 800), legend=:topleft);

for (idx, x_val) in enumerate(x_fixed_values)
    
    # Prepare prediction inputs
    X_test_slice = zeros(2, length(z_plot_grid))
    X_test_slice[1, :] .= BOOP.rescale(x_val, [-1.], [1.]; integ=0)
    X_test_slice[2, :] .= z_plot_grid

    # Predict
    μScaled, σ²Scaled = predict_f(gp, X_test_slice)
    μ = μScaled*σ_y .+ μ_y
    σ = sqrt.(σ²Scaled)*σ_y
    # 2xSE bands
    lower = μ .- 2 .* σ
    upper = μ .+ 2 .* σ

    # Plot GP (staircases)
    plot!(p_steps, z_plot_grid, μ, 
          ribbon=(μ .- lower, upper .- μ),
          fillalpha=0.2,
          lw=2,
          label="GP (Trappsteg)",
          title="Fixerat x = $(x_val)",
          xlabel="Diskret variabel (z)",
          ylabel="Output",
          subplot=idx,
          color=idx
    )

    # Plot truth.
    y_true = [true_function([x_val, z]) for z in z_plot_grid]
    plot!(p_steps, z_plot_grid, y_true, 
          label="Sann funktion", 
          linestyle=:dash, 
          color=:black, 
          alpha=0.6,
          subplot=idx
    )

    # Add scatter of data points near this x-value.
    mask = abs.(X_final'[1, :] .- x_val) .< 0.05
    
    if any(mask)
        scatter!(p_steps, X_final'[2, mask], y_final[mask],
                 label="Data (nära x=$(x_val))",
                 markercolor=:red,
                 markersize=5,
                 subplot=idx
        )
    end
end;


pHeat = plot(p1, p2);
pSlice = plot(slice_plots..., layout=(1,3));
plot(pHeat,pSlice,p3,layout=(3,1), size=(1500,1200))