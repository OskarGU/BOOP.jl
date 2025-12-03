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
    noise = 0.5 * randn()
    
    # Scale up a bit.
    return 10.0 * disc_part * cont_part + noise
end

# No noise version for plotting
f_true_plot(x, z) = 10.0 * exp(-(round(z) - 7)^2 / 8.0) * sqrt(abs(x))*cos(3 * (x*1.3 - 0.5)) 

# Optimization bounds.
x_lo, x_hi = -1.0, 1.0
z_lo, z_hi = 2, 12
d = 2

# Generate starting values
N_init = 2
X_warm = zeros(N_init, d)
#X_warm[1, :] = [-0.95 7.]
#X_warm[2, :] = [-0.65,5.]
X_warm[:, 1] = rand(Uniform(x_lo, x_hi), N_init)   
X_warm[:, 2] = rand(z_lo:z_hi, N_init)              
y_warm = [true_function(row) for row in eachrow(X_warm)]

warmStart = (X_warm, y_warm)
hcat(X_warm, y_warm)
# ==============================================================================
# BO SETTINGS
# ==============================================================================

# Set up the kernel with Garrido-Merchán adjustments
# Use Matern52Ard ase base-kernel.
baseKernel = Mat52Ard([0.1;0.6], 0.8) 
GMKernel = BOOP.GarridoMerchanKernel(baseKernel, [2], [z_lo:z_hi])


modelSettings = (        
    kernelBounds = [[-3.0, 0.5, -2.0], [3.0, 3.0, 5.0]], 
    noiseBounds = [-3.0, 2.0],
    # Bounds for rescaleing.
    xBounds = ([[x_lo], [x_hi]]) 
)

optSettings = OptimizationSettings(
    nIter = 2,          
    n_restarts = 10,
    acq_config = EIConfig(ξ=0.11) 
    #acq_config = UCBConfig(κ=2.5)
)


#============================================================#
# Set some priors on the kernel hyperparameters.
#=====================================================#

# 

# Since GaussianProcesses.jl optimizes in log-space, we use Normal priors.
# Normal(μ, σ) on log(θ) corresponds to LogNormal on θ.

# Prior for Continuous Variable (x1)
# Range is [-1, 1] (width 2). We want a length scale around 0.5.
# Normal(0.0, 1.0) => Median length scale exp(0) = 1.0.
prior_cont = Normal(-0.5, 0.5)

# Prior for Discrete Variable (x2)
# Step size is 1.0. To avoid the "Independence Trap" (where the GP treats neighbors
# as unrelated), the length scale MUST be > 1.0.
# We aim for a length scale around 3.0 (exp(1.1)).
# We set mean to 1.0 and a tighter std (0.5) to push the optimizer away from 0.
prior_disc = Normal(1.0, 0.5)

# Prior for Signal Variance
prior_sig = Normal(0.0, 1.0)

# Apply to Kernel
# Parameter order for Mat52Ard (dim=2): [log(ℓ_x1), log(ℓ_x2), log(σ_f)]
priors = [prior_cont, prior_disc, prior_sig]

set_priors!(GMKernel, priors)

gp_template = GPE(
    X_warm',              # Input data (transponerat för GPE)
    y_warm,               # Output data
    MeanConst(0.0),       # Målfunktionens medelvärde (för standardiserad data)
    GMKernel,            # Kerneln vi byggde ovan (kopieras inuti BO)
    -1.0                  # Startvärde för logNoise
)
# ==============================================================================
# Run BAYESIAN OPTIMIZATION
# ==============================================================================


# With prior, optimization gets way faster when adding this curvature! 
warmStart = (X_final, y_final)
@time gp, X_final, y_final, max_x, max_val, max_obs_x, max_obs_val = BO(
    true_function, gp_template, modelSettings, optSettings, warmStart; DiscreteKern=1
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
x_fixed_values = [-0.6, 0.9, 0.5]

# Layout 2x2
plot_layout = @layout [a b c]
pSteps = plot(layout=plot_layout, size=(1000, 800), legend=:topleft);

for (idx, x_val) in enumerate(x_fixed_values)
    
    # Prepare prediction inputs
    X_test_slice = zeros(2, length(z_plot_grid))
    X_test_slice[1, :] .= rescale([x_val], [-1.], [1.]; integ=[0])
    X_test_slice[2, :] .= z_plot_grid

    # Predict
    μScaled, σ²Scaled = predict_f(gp, X_test_slice)
    μ = μScaled*σ_y .+ μ_y
    σ = sqrt.(σ²Scaled)*σ_y
    # 2xSE bands
    lower = μ .- 2 .* σ
    upper = μ .+ 2 .* σ

    # Plot GP (staircases)
    plot!(pSteps, z_plot_grid, μ, 
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
    y_true = [f_true_plot(x_val, z) for z in z_plot_grid]
    plot!(pSteps, z_plot_grid, y_true, 
          label="Sann funktion", 
          linestyle=:dash, 
          color=:black, 
          alpha=0.6,
          subplot=idx
    )

    # Add scatter of data points near this x-value.
    mask = abs.(X_final'[1, :] .- x_val) .< 0.05
    
    if any(mask)
        scatter!(pSteps, X_final'[2, mask], y_final[mask],
                 label="Data (nära x=$(x_val))",
                 markercolor=:red,
                 markersize=5,
                 subplot=idx
        )
    end
end;


pHeat = plot(p1, p2);
pSlice = plot(slice_plots..., layout=(1,3));
len=(length(X_final[:,1]))
plot(pHeat,pSlice,pSteps,layout=(3,1), size=(1500,1200),plot_title="Iteration: $len")








###############################
# Skapa och spara en animation av optimeringsprocessen
###############################
using Plots, Statistics, Printf

# ==============================================================================
# 1. SETUP
# ==============================================================================
warmStart = (X_warm, y_warm) # Startvärden från tidigare
# Initiera warmStart om det behövs
if !@isdefined(warmStart)
    println("Initierar warmStart...")
    N_init = 5
    X_init = zeros(N_init, 2)
    X_init[:, 1] = rand(x_lo[1]:0.01:x_hi[1], N_init)
    X_init[:, 2] = rand(z_lo:z_hi, N_init)
    y_init = [true_function(row) for row in eachrow(X_init)]
    warmStart = (X_init, y_init)
end

# Settings för ETT steg i taget
step_opt_settings = OptimizationSettings(
    nIter = 1,
    n_restarts = 5,
    acq_config = EIConfig(ξ=0.1) 
)

frames = 20

# ==============================================================================
# 2. ANIMATIONSLOOP
# ==============================================================================
anim = @animate for iter in 1:frames
    println("Genererar frame $iter / $frames ...")

    # --- A. KÖR OPTIMERING (1 STEG) ---
    global gp, X_final, y_final, max_x, max_val, _, _ = BO(
        true_function, gp_template, modelSettings, step_opt_settings, warmStart; 
        DiscreteKern=[2] 
    )
    
    global warmStart = (X_final, y_final)
    μ_y_local, σ_y_local = mean(y_final), std(y_final)

    # --- B. SKAPA PLOTTARNA ---

    # 1. Heatmaps
    Z_model = zeros(length(zs), length(xs))
    for (r, z) in enumerate(zs), (c, x) in enumerate(xs)
        x_sc = 2 * (x - x_lo) / (x_hi - x_lo) - 1
        pt = reshape([x_sc, Float64(z)], 2, 1) 
        μ_sc, _ = predict_f(gp, pt)
        Z_model[r, c] = μ_sc[1] * σ_y_local + μ_y_local
    end

    p1 = heatmap(xs, zs, Z_true, title="Truth", xlabel="x", ylabel="z", c=:viridis)
    p2 = heatmap(xs, zs, Z_model, title="GP Model", xlabel="x", ylabel="z", c=:viridis)
    scatter!(p2, X_final[:,1], X_final[:,2], label="", mc=:red, ms=5, ma=0.8)
    pHeat = plot(p1, p2, layout=(1,2), size=(800, 300))

    # 2. Slice Plots (Z-snitt)
    slice_plots_vec = []
    for z_val in [5, 7, 8]
        y_true_slice = [f_true_plot(x, z_val) for x in xs]
        y_pred_slice = zeros(length(xs))
        sig_pred_slice = zeros(length(xs))
        
        for (k, x) in enumerate(xs)
            x_sc = 2 * (x - x_lo) / (x_hi - x_lo) - 1
            pt = reshape([x_sc, Float64(z_val)], 2, 1)
            μ, σ2 = predict_f(gp, pt)
            y_pred_slice[k] = μ[1] * σ_y_local + μ_y_local
            sig_pred_slice[k] = sqrt(max(σ2[1], 0)) * σ_y_local
        end
        
        p = plot(xs, y_true_slice, label="Sanning", lw=1.5, lc=:black, ls=:dash, title="z = $z_val", legend=false)
        plot!(p, xs, y_pred_slice, label="GP", lw=2, lc=:blue, ribbon=(2*sig_pred_slice, 2*sig_pred_slice), fillalpha=0.2, fc=:blue)
        
        mask = (round.(X_final[:, 2]) .== z_val)
        if any(mask)
            scatter!(p, X_final[mask, 1], y_final[mask], mc=:red, ms=6)
        end
        push!(slice_plots_vec, p)
    end
    pSlice = plot(slice_plots_vec..., layout=(1,3))

    # 3. STEP PLOTS (Här är färgfixen!)
    z_cont_grid = range(z_lo, z_hi, length=400)
    x_fixed_vals = [-0.6, 0.5, 0.6] 
    step_plots_vec = []

    for (k, x_fix) in enumerate(x_fixed_vals)
        y_true_step = [f_true_plot(x_fix, z) for z in z_cont_grid]
        y_pred_step = zeros(length(z_cont_grid))
        sig_pred_step = zeros(length(z_cont_grid))
        
        x_sc = 2 * (x_fix - x_lo) / (x_hi - x_lo) - 1
        
        for (k, z) in enumerate(z_cont_grid)
            pt = reshape([x_sc, z], 2, 1)
            μ, σ2 = predict_f(gp, pt)
            y_pred_step[k] = μ[1] * σ_y_local + μ_y_local
            sig_pred_step[k] = sqrt(max(σ2[1], 0)) * σ_y_local
        end
        
        # --- PLOTTNING MED RÄTT FÄRGER ---
        p = plot(z_cont_grid, y_pred_step, 
            title="x = $x_fix", xlabel="z", legend=false, 
            lw=2, lc=k,                 # Blå Linje
            ribbon=(2*sig_pred_step, 2*sig_pred_step), # Osäkerhetsband
            fillalpha=0.2, fc=k         # Ljusblå fyllning
        )
        
        # Sanningen (Svart streckad)
        plot!(p, z_cont_grid, y_true_step, lc=:black)
        
        # Röda Prickar (Data nära detta x)
        mask_x = abs.(X_final[:, 1] .- x_fix) .< 0.1
        if any(mask_x)
            scatter!(p, X_final[mask_x, 2], y_final[mask_x], mc=:red, ms=5)
        end
        push!(step_plots_vec, p)
    end
    
    pSteps = plot(step_plots_vec..., layout=(1,3))

    # 4. SLUTMONTERING
    N_total = size(X_final, 1)
    
    # Använd layout med korrekta höjder (som summerar till 1.0)
    final_plot = plot(pHeat, pSlice, pSteps, 
        layout = grid(3, 1, heights=[0.33, 0.33, 0.34]),
        size = (1200, 1200),
        plot_title = "GM-BO Optization: Iteration $iter (Total $N_total evaluations)",
        plot_titlefontsize = 16,
        left_margin = 5Plots.mm
    )
    
    final_plot
end

# SPARA
gg = gif(anim, "Full_GM_Animation_Blue.gif", fps=1)
pwd()