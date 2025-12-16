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
    noise = 0.55 * randn()
    
    # Scale up a bit.
    return 10.0 * disc_part * cont_part + noise
end

# No noise version for plotting
f_true_plot(x, z) = 10.0 * exp(-(round(z) - 7)^2 / 8.0) * sqrt(abs(x))*cos(3 * (x*1.3 - 0.5)) 

# Optimization bounds.
x_lo, x_hi = [-1.0; 2.0], [1.0; 12.0]
d = 2

# Generate starting values
N_init = 2
X_warm = zeros(N_init, d)
#X_warm[1, :] = [-0.95 7.]
#X_warm[2, :] = [-0.65,5.]
X_warm[:, 1] = rand(Uniform(x_lo[1], x_hi[1]), N_init)   
X_warm[:, 2] = rand(Uniform(x_lo[2], x_hi[2]), N_init)               
y_warm = [true_function(row) for row in eachrow(X_warm)]

warmStart = (X_warm, y_warm)
hcat(X_warm, y_warm)
# ==============================================================================
# BO SETTINGS
# ==============================================================================

# Set up the kernel with Garrido-Merchán adjustments
# Use Matern52Ard ase base-kernel.
baseKernel = Mat52Ard([0.1;0.6], 0.8) 


modelSettings = (           
    kernelBounds = [[-3.0, 0.5, -2.0], [3.0, 3.0, 5.0]], 
    noiseBounds = [-3.0, 2.0],
    # Bounds for rescaleing.
    xBounds = ([x_lo, x_hi]) 
)

optSettings = OptimizationSettings(
    nIter = 3,          
    n_restarts = 10,
    acq_config = EIConfig(ξ=0.31) 
    #acq_config = UCBConfig(κ=2.5)
    #acq_config = KGHConfig(n_z=50)
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

# Prior for Signal Variance
prior_sig = Normal(0.0, 1.0)

# Apply to Kernel
# Parameter order for Mat52Ard (dim=2): [log(ℓ_x1), log(ℓ_x2), log(σ_f)]
priors = [prior_cont, prior_cont, prior_sig]

set_priors!(baseKernel, priors)

gp_template = GPE(
    X_warm',              # Input data (transponerat för GPE)
    y_warm,               # Output data
    MeanConst(mean(y_warm)),       # Målfunktionens medelvärde (för standardiserad data)
    baseKernel,            # Kerneln vi byggde ovan (kopieras inuti BO)
    -1.0                  # Startvärde för logNoise
)

# ==============================================================================
# Run BAYESIAN OPTIMIZATION
# ==============================================================================

# IMPORTANT: DiscreteKern=1 makes sure that the last dimension does not rescale.
warmStart = (X_final, y_final)
@time gp, X_final, y_final, max_x, max_val, max_obs_x, max_obs_val = BO(
    true_function, modelSettings, optSettings, warmStart; DiscreteKern=1
)

# With prior, optimization gets way faster when adding this curvature! 
warmStart = (X_final, y_final)
@time gp, X_final, y_final, max_x, max_val, max_obs_x, max_obs_val = BO(
    true_function, gp_template, modelSettings, optSettings, warmStart; DiscreteKern=0
)


println("\nGlobal Model Max hittat vid: $max_x")
println("Värde vid max: $(round(max_val, digits=3))")

# ==============================================================================
# PLOTTING RESULTS
# ==============================================================================

# Grids
xs1 = range(x_lo[1], x_hi[1], length=100)
xs2 = range(x_lo[2], x_hi[2], length=100)


# Colect mean and std of final observations for rescaling plots
μy, σy = mean(y_final), std(y_final)

# Evaluate true function on the grid.
Z_true = [f_true_plot(x1, x2) for x1 in xs1, x2 in xs2] 

p1 = heatmap(xs1, xs2, Z_true', title="Sanning", xlabel="x (Kont)", ylabel="z (Diskret)", c=:viridis);
p2 = heatmap(gp, title="GP Modell", xlabel="x (Kont)", ylabel="z (Diskret)", c=:viridis);
# Add sampled points.
XFinalScaled = rescale(X_final, x_lo, x_hi)
scatter!(p2, XFinalScaled[:,1], XFinalScaled[:,2], label="Sampel", mc=:red, ms=4, legend=false);
plot(p1,p2, size=(1400,400))




# Slice plots at specific discrete z-values
vals = [5, 7, 8]

z_scaled = 2 .* (vals .- x_lo[2]) ./ (x_hi[2] - x_lo[2]) .- 1
zSlices = round.(z_scaled, digits=2)
slicePlots = []

function findClosest(vec, target) 
    # 1. Beräkna det absoluta avståndet mellan varje element och target (5)
    #    Använder broadcasting (punkt-notation) för att göra det effektivt.
    distances = abs.(vec .- target)
    
    # 2. Hitta indexet för det minsta avståndet
    #    argmin() returnerar indexet för det minsta värdet i vektorn.
    closest_index = argmin(distances)
    
    return closest_index
end

for zVal in zSlices
    pts = hcat(collect(xs1), fill(float(zVal), length(xs1)))'
    
    meanPredScaled, varPredScaled = predict_f(gp, pts)
    stdPredScaled = sqrt.(max.(varPredScaled, 0.0))
    
    ySliceModel = meanPredScaled .* σy .+ μy
    stdSliceModel = stdPredScaled .* σy
    
    zOrig = (zVal + 1) / 2 * (x_hi[2] - x_lo[2]) + x_lo[2]
    
    isAtZ = (round.(X_final[:, 2]) .== round(zOrig))
    samplesX = X_final[isAtZ, 1]
    samplesY = y_final[isAtZ]

    idxTrue = argmin(abs.(collect(xs2) .- zOrig))

    p = plot(xs1, Z_true[:, idxTrue], 
             label="Sanning (z=$(round(zOrig, digits=1)))", 
             lw=2, lc=:black, 
             title="Snitt vid z ≈ $(round(zOrig, digits=1))", 
             ylims=(-15, 15))
             
    plot!(p, xs1, ySliceModel, label="GP Mean", lw=2, lc=:blue, 
          ribbon=(1.96 * stdSliceModel, 1.96 * stdSliceModel), fillalpha=0.2, fc=:blue)
          
    scatter!(p, samplesX, samplesY, label="Observerat", mc=:red, ms=6)
    
    push!(slicePlots, p)
end

# Combine the subplots
l = @layout [a b; c d e]
final_plot = plot(p1, p2, slicePlots..., layout=l, size=(1200, 1000));



# Vary z continuously to see the step behavior
z_plot_grid = rescale(collect(range(0, 13, length=100)), [x_lo[2]], [x_hi[2]])

# Select 4 different fixed x-values to plot the steps at.
x_fixed_values = [-0.6, 0.9, 0.5]

# Layout 2x2
plot_layout = @layout [a b c]
pSteps = plot(layout=plot_layout, size=(1000, 800), legend=:topleft);

for (idx, x_val) in enumerate(x_fixed_values)
    
    # Prepare prediction inputs
    X_test_slice = zeros(2, length(z_plot_grid))
    X_test_slice[1, :] .= BOOP.rescale([x_val], [-1.], [1.]; integ=0)
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
 

    plot!(pSteps, z_plot_grid, Z_true[findClosest(collect(xs1), inv_rescale([x_val], [x_lo[1]], [x_hi[1]])),:], 
          label="Sann funktion", 
          linestyle=:dash, 
          color=:black, 
          alpha=0.6,
          subplot=idx
    )

    # Add scatter of data points near this x-value.
    mask = abs.(rescale(X_final'[1, :], [x_lo[1]], [x_hi[1]]) .- x_val) .< 0.05
    
    if any(mask)
        scatter!(pSteps, rescale(X_final'[2, mask], [x_lo[2]], [x_hi[2]]), y_final[mask],
                 label="Data (nära x=$(x_val))",
                 markercolor=:red,
                 markersize=5,
                 subplot=idx
        )
    end
end;


pHeat = plot(p1, p2);
pSlice = plot(slicePlots..., layout=(1,3));
plot(pHeat,pSlice,pSteps,layout=(3,1), size=(1500,1200))