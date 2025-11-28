using GaussianProcesses
using Random, Optim, Distributions, Plots, Statistics, LinearAlgebra, Measures
using BOOP 

Random.seed!(42)

# ==============================================================================
# DEFINE THE TARGET FUNCTION
# The idea is that we have something similar to a VAR model where we have three
# hyperparameters to tune.
# ==============================================================================
# Variables:
# x[1] (Continuous): Overall Shrinkage (λ1). Range [0.0, 1.0]. (0=hard, 1=loose)
# x[2] (Kontinuerlig): Lag Decay (λ2). Range [0.0, 5.0]. (High = fast decay)
# x[3] (Discrete):      number of lags (p). Range [1, 8].

# Target function is with noise
function true_var_proxy(x_vec)
    # Variabler
    x1 = x_vec[1]      
    x2 = x_vec[2]       
    z  = round(x_vec[3]) # Lags [1, 8]

    # 1. Z-score tops around z=5
    score_z = -0.5 * (z - 5)^2 + 6.0

    # 2. Banana-shape
    # The more lags (z), the lower (harder) the base-nivån, x1.
    target_balance = 0.5 - 0.05 * z 
    
    # Balance: compare x1 to x2. 
    current_balance = x1 - 0.002 * x2^4
    banana_penalty = -150.0 * (current_balance - target_balance)^2

    # 3. Weak centering of decay around 2.5
    decay_centering = -0.1 * (x2 - 2.5)^2

    # 4. Decay should not be too small
    zero_decay_penalty = -12.0 * exp(-4.0 * x2)

    # Sum of the parts and noise
    true_val = score_z + banana_penalty + decay_centering + zero_decay_penalty
    noise = 0.1 * randn()
    return max(true_val + noise, -50.0)
end

# Same as above but noise-free, for plotting.
function f_true_plot(x1, x2, z)
    score_z = -0.5 * (z - 5)^2 + 6.0
    target_balance = 0.5 - 0.05 * z 
    current_balance = x1 - 0.002 * x2^4
    banana_penalty = -150.0 * (current_balance - target_balance)^2
    decay_centering = -0.1 * (x2 - 2.5)^2
    zero_decay_penalty = -12.0 * exp(-4.0 * x2)
    return score_z + banana_penalty + decay_centering + zero_decay_penalty
end

# ==============================================================================
# Specify settings for the Bayesian optimization

# Bounds 
x1_lo, x1_hi = 0.0, 1.    # Overall Shrinkage
x2_lo, x2_hi = 0.0, 5.0  # Lag Decay
z_lo, z_hi   = 1, 8      # number of lags
d = 3

# Generate some initial data for warm start.
N_init = 5
X_warm = zeros(N_init, d)
X_warm[:, 1] = rand(Uniform(x1_lo, x1_hi), N_init)
X_warm[:, 2] = rand(Uniform(x2_lo, x2_hi), N_init)
X_warm[:, 3] = rand(z_lo:z_hi, N_init) # Heltal!

y_warm = [true_var_proxy(row) for row in eachrow(X_warm)]
warmStart = (X_warm, y_warm)

# Specify the kernel, model settings, and optimization settings

# 1. Kernel (3 dimensions: 2 continuous + 1 discrete)
# We work with a Matérn 5/2 kernel with ARD as a base here.
startLogℓ = [-0.5, -0.5, 1.0] 
startSignalσ² = 1.0 
baseKernel = Mat52Ard(startLogℓ, log(startSignalσ²)) 

# G-M Kernel: Tells the kernel that the third variable is discrete.
GMKernel = BOOP.GarridoMerchanKernel(baseKernel, [3], [z_lo:z_hi])


modelSettings = (
    mean = MeanConst(mean(y_warm)),
    kernel = deepcopy(gm_kernel),   
    logNoise = -1.0,                
    
    # Important!: put bounds on the discrete dimensions so that the length scale 
    # is not too short! (< 0.5).
    # Bounds ordning: [ℓx1, ℓx2, ℓz, signalσ²]
    kernelBounds = [[-2.0, -2.0, 0.5, -2.0], [3.0, 3.0, 4.0, 5.0]], 
    noiseBounds = [-2.0, 2.0],
    xdim = d,
    
    # Bounds for continuous variables (used by rescale())
    xBounds = ([x1_lo, x2_lo], [x1_hi, x2_hi]) 
)


opt_settings = OptimizationSettings(
    nIter = 3,           
    n_restarts = 15,
    acq_config = EIConfig(ξ=.1) 
)


# Priors:

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
priors = [prior_cont, prior_cont, prior_disc, prior_sig]

set_priors!(GMKernel, priors)

gp_template = GPE(
    X_warm',              # Input data (transponerat för GPE)
    y_warm,               # Output data
    MeanConst(mean(y_final)),       # Målfunktionens medelvärde (för standardiserad data)
    GMKernel,            # Kerneln vi byggde ovan (kopieras inuti BO)
    -1.0                  # Startvärde för logNoise
)

# ==============================================================================
# RUN THE BAYESIAN OPTIMIZATION
# ==============================================================================

# DiscreteKern=1 means that last dimension is discrete.
# We use the warm start data defined above.
warmStart = (X_final, y_final)
#warmStart = (X_warm, y_warm)
@time gp, X_final, y_final, max_x, max_val, max_obs_x, max_obs_val = BO(
    true_var_proxy, modelSettings, opt_settings, warmStart; DiscreteKern=1
)


# With prior
warmStart = (X_final, y_final)
@time gp, X_final, y_final, max_x, max_val, max_obs_x, max_obs_val = BO(
    true_var_proxy, gp_template, modelSettings, opt_settings, warmStart; DiscreteKern=1
)

println("\nGlobalt Optimum found:")
println("  Overall Shrinkage (x1): $(round(max_x[1], digits=3))")
println("  Lag Decay (x2):         $(round(max_x[2], digits=3))")
println("  Number of lags (z):       $(round(max_x[3]))")
println("  Estimated LML:          $(round(max_val, digits=3))")

# ==============================================================================
# PLOTTING
# ==============================================================================

# We start by fixing x (number of lags) for 3 different levels:
z_levels = [3, 5, 7]

# Containers for true heatmaps and the GP model heatmaps.
plots_true = []
plots_model = []

# Grid for x1 and x2
x1_grid = range(x1_lo, x1_hi, length=50)
x2_grid = range(x2_lo, x2_hi, length=50)

# Collect information for rescaling
μ_y, σ_y = mean(y_final), std(y_final)

# Constructiung the heatmaps.
for z_val in z_levels
    # Containers for functions values.
    Z_true_mat = zeros(length(x2_grid), length(x1_grid))
    Z_pred_mat = zeros(length(x2_grid), length(x1_grid))
    
    # Compute function values for the different combinations 
    for (i, x1) in enumerate(x1_grid)
        for (j, x2) in enumerate(x2_grid)
            Z_true_mat[j, i] = f_true_plot(x1, x2, z_val)
            
            # The model
            # Scale x1 and x2 to [-1, 1] for GP-prediction
            x1_s = 2 * (x1 - x1_lo) / (x1_hi - x1_lo) - 1
            x2_s = 2 * (x2 - x2_lo) / (x2_hi - x2_lo) - 1
            
            # Input to GP: [x1_scaled, x2_scaled, z_raw]
            pt = [x1_s, x2_s, Float64(z_val)]
            
            μ_sc, _ = predict_f(gp, reshape(pt, d, 1))
            Z_pred_mat[j, i] = μ_sc[1] * σ_y + μ_y
        end
    end
    
    # Common color limits for better comparison
    common_clim = (-8, 6.5) 

    # Heatmap ove true function
    p_true = heatmap(x1_grid, x2_grid, Z_true_mat, 
                title="Truth (z=$z_val)",
                xlabel="Shrinkage (x1)", 
                ylabel="Decay (x2)",
                c=:viridis, clim=common_clim
    )
    push!(plots_true, p_true)

    # GP heatmap + sampled points
    mask = abs.(X_final[:, 3] .- z_val) .< 0.1
    x1_samples = X_final[mask, 1]
    x2_samples = X_final[mask, 2]
    
    p_model = heatmap(x1_grid, x2_grid, Z_pred_mat, 
                title="Model (z=$z_val)",
                xlabel="Shrinkage (x1)", 
                ylabel="Decay (x2)",
                c=:viridis, clim=common_clim
    )
    scatter!(p_model, x1_samples, x2_samples, label="Sampel", mc=:red, ms=5, legend=false)
    push!(plots_model, p_model)
end

# Combine the plots into one
heatmapTrue = plot(plots_true..., layout=(1, length(z_levels)), size=(1200, 300), cbar=false,
    margin=1mm,            
    left_margin=10mm,       
    bottom_margin=10mm
);

heatmapGP = plot(plots_model..., layout=(1, length(z_levels)), size=(1200, 300), cbar=false,    
    margin=1mm,           
    left_margin=10mm,      
    bottom_margin=10mm
);    

final_heatmap_plot = plot(heatmapTrue, heatmapGP, layout=(2,1), size=(1200, 600))




# --- DEL 2: SNITT (Slices) ---
# Låt oss fixera Overall Shrinkage (x1) till ett "vettigt" värde (t.ex. 0.2)
# och se hur Lag Decay (x2) interagerar med antalet laggar.
x1_fixar = [0.1; 0.7; 0.4]
x2_grid = range(x2_lo, x2_hi-0.75, length=50) # shorter to make nicer plots.
slice_plots = []
i =0
for z_val in z_levels
    i += 1
    x1_fix = x1_fixar[i]
    y_true_slice = [f_true_plot(x1_fix, x2, z_val) for x2 in x2_grid]
    
    # Modell prediktion
    x1_s = 2 * (x1_fix - x1_lo) / (x1_hi - x1_lo) - 1
    x2_s_vec = 2 .* (collect(x2_grid) .- x2_lo) ./ (x2_hi - x2_lo) .- 1
    
    # Bygg matris: 3 x N
    pts = vcat(fill(x1_s, length(x2_grid))', x2_s_vec', fill(float(z_val), length(x2_grid))')
    
    μ_sc, σ2_sc = predict_f(gp, pts)
    μ_y, σ_y = mean(y_final), std(y_final)
    
    y_pred = μ_sc .* σ_y .+ μ_y
    σ_pred = sqrt.(max.(σ2_sc, 0.0)) .* σ_y
    
    p = plot(x2_grid, y_true_slice, label="Sanning", lw=2, color=:black, 
             title="Snitt: x1=$x1_fix, z=$z_val", xlabel="Lag Decay (x2)", ylabel="LML")
    
    plot!(p, x2_grid, y_pred, label="GP Mean", lw=2, color=:blue, 
          ribbon=(1.96*σ_pred, 1.96*σ_pred), fillalpha=0.2)
   
    push!(slice_plots, p)
end

# Sammanställ allt
pSlice = plot(slice_plots..., layout=(1, length(z_levels)), size=(800, 200));

all_plots = [plots_true...; plots_model...; slice_plots...]

# 2. Plotta med 3x3 layout
final_grid_plot = plot(all_plots..., 
    layout = (3, 3),          # 3 Rader, 3 Kolumner
    size = (1200, 1000),      # Rejäl storlek (höjden viktig för 3 rader)
    
    # Marginaler och typsnitt
    margin = 1mm,             # VIKTIGT: Ger luft mellan graferna
    left_margin = 1mm,       # Plats för Y-axeltitlar
    bottom_margin = 3mm,     # Plats för X-axeltitlar
    
    titlefontsize = 10,
    guidefontsize = 8,
    tickfontsize = 8,
    
    #plot_title = "Bayesian Optimization Resultat (3D)",
    plot_titlefontsize = 16,
    cbar=false
)


#######################
# Different z-slices #

# 1. Define scenarios to visualize
# We fix x1 and x2 to specific values to see the cross-section over Z
fixed_scenarios = [
    (name="Optimal shrinkage", coords=max_x[1:2]), 
    (name="Tight overall/Low decay", coords=[0.1, 0.3]),    # Low shrinkage, Low decay
    (name="Loose overall/fast decay", coords=[0.9, 4.5])      # High shrinkage, High decay
]

# Grids
z_grid_fine = range(z_lo, z_hi, length=200)
z_integers = z_lo:z_hi
step_plots = []

# Stats for unscaling
μ_y, σ_y = mean(y_final), std(y_final)

for scenario in fixed_scenarios
    x1_val, x2_val = scenario.coords
    
    # Prediction over Z
    x1_s = 2 * (x1_val - x1_lo) / (x1_hi - x1_lo) - 1
    x2_s = 2 * (x2_val - x2_lo) / (x2_hi - x2_lo) - 1
    
    X_pred = zeros(d, length(z_grid_fine))
    X_pred[1, :] .= x1_s
    X_pred[2, :] .= x2_s
    X_pred[3, :] .= z_grid_fine 
    
    μ_sc, σ2_sc = predict_f(gp, X_pred)
    y_pred = μ_sc .* σ_y .+ μ_y
    σ_pred = sqrt.(max.(σ2_sc, 0.0)) .* σ_y
    
    # Tru function
    y_true_int = [true_var_proxy([x1_val, x2_val, z]) for z in z_integers]

    # Make plot
    title_str = "$(scenario.name)\n(x1=$(round(x1_val, digits=2)), x2=$(round(x2_val, digits=2)))"
    
    p = plot(title = title_str,
             xlabel = "Number of Lags (z)",
             ylabel = "LML Score",
             legend = :bottomright,
             grid = true,
             ylim = (-40, 10)) 
    
    # 1. GP Mean + Uncertainty
    plot!(p, z_grid_fine, y_pred, 
          label = "GP Mean (±2 SE)", 
          lw = 3, 
          color = :blue, 
          ribbon = (1.96 .* σ_pred, 1.96 .* σ_pred), 
          fillalpha = 0.2)

    # 2. True Function
    plot!(p, z_integers, y_true_int, 
          label = "True Function", 
          lw = 2, 
          linestyle = :solid, 
          color = :black)
    
    push!(step_plots, p)
end

# Combine plots
final_simple_plot = plot(step_plots..., 
    layout = (1, 3), 
    size = (1200, 400),
    margin = 8mm
)

