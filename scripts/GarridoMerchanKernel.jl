using GaussianProcesses
using BOOP
using Plots, Random, LinearAlgebra, Statistics, Distributions

# ==============================================================================
# Example GARRIDO-MERCHÁN KERNEL for GPs
# ==============================================================================


# ==============================================================================
# Simulate some data with one discrete and one continuous input.
# ==============================================================================
Random.seed!(42)

function true_function(x_vec)
    x_cont = x_vec[1]
    z_int  = round(x_vec[2]) 
    return sin(x_cont * 2) * (z_int^2 / 1) + 0.05 * (z_int - 5)^2
end

N = 35
X_train = zeros(2, N)

# Continuous
X_train[1, :] = rand(N) .* 5.0;

# Discrete
steps=4
X_train[2, :] = rand(1:steps, N);

# Y-data med lite brus
y_train = [true_function(X_train[:, i]) for i in 1:N];
y_train .+= 0.5 .* randn(N);





# ==============================================================================
# Select kernel and mean.
# ==============================================================================

# We use Matern 5/2 ARD.
# Det betyder att vi har en längdskala för dim 1 och en för dim 2.

startLogℓ = [0.5, 0.20]
base_k = Matern(5/2, startLogℓ, 2.0)

# Define our kernel for mixed continuous discrete, we give it the base kernel and tell it that the second input is discrete.
gm_kernel = BOOP.GarridoMerchanKernel(base_k, [2], [1:16])

# Fit the GP with the starting parameters.
gp = GP(X_train, y_train, MeanZero(), gm_kernel)


# --- Set priors for GP log-parameters ---
# Note that it is the log of the scales!

# Prior for continuous
priorContinuous_ℓ = Normal(log(.5), 1.5)

# Prior for discrete variable.
# Note that it is not scaled so some care should be given.
priorDisc_ℓ = Normal(log(1.5), 1.5)

# Prior for signal variance
priorProcess_σ = Normal(1.0, 3.)

# Prior for measurement noise
priorNoise_σ = Normal(log(0.1), 3.0)

# Apply the prior to the GP object
set_priors!(gp.logNoise, [priorNoise_σ])

set_priors!(gp.kernel, [priorContinuous_ℓ, priorDisc_ℓ, priorProcess_σ])


# Define some search boundaries for the optimizer, we don't want it to go completely crazy.

# Kernel lower bounds (log-scale)
lb_kern = log.([0.01, 0.5, 0.1]) 

# Kernel upper bounds (log-scale)
ub_kern = log.([10.0, 10.0, 10.0])

kern_bounds = [lb_kern, ub_kern]

# Same for noise bounds
lb_noise = log.([0.1])
ub_noise = log.([5.0])  # Max brus

noise_bounds = [lb_noise, ub_noise]

println("Training GP ...")
optOptions = Optim.Options(time_limit = 1.0, show_trace = true)
@time optimize!(gp, kernbounds=kern_bounds, noisebounds=noise_bounds, options=optOptions)

println("Length scales: ", exp.(get_params(gp.kernel)[1:2]))
println("Process variance: ", exp.(get_params(gp.kernel)[3]))
println("Measurement noise: ", exp.(get_params(gp.logNoise)))


# ==============================================================================
# 4. VISUALISERA
# ==============================================================================
# Nu är plottningen superenkel eftersom vi jobbar i originalskala direkt.

x_grid = range(-1, 5, length=50)
z_grid = range(1, 5, length=100) # Fint grid för att se stegen

Z_pred = zeros(length(x_grid), length(z_grid))

for (i, x) in enumerate(x_grid)
    for (j, z) in enumerate(z_grid)
        input_point = reshape([x; z], 2, 1)
        μ, _ = predict_y(gp, input_point)
        Z_pred[i, j] = μ[1]
    end
end

heatmap(x_grid, z_grid, Z_pred',
    xlabel="Kontinuerlig (x)",
    ylabel="Heltal (z, 1-10)",
    title="GP med blandade skalor (ARD)",
    c=:viridis
)






# Värden att plotta den kontinuerliga variabeln (x) över
x_plot_grid = range(0.05, 5, length=100)

# De diskreta "trappstegen" vi vill fixera (z)
z_fixed_values = [1, 3, 6, 4]

# Skapa en layout för 4 subplots (2x2)
plot_layout = @layout [a b; c d]
p_combined = plot(layout=plot_layout, size=(1000, 800), legend=:topleft)


for (idx, z_val) in enumerate(z_fixed_values)
    # Skapa testpunkter för prediktion: (x_plot_grid, z_val)
    X_test_slice = hcat([reshape([x_val; z_val], 2, 1) for x_val in x_plot_grid]...)

    # Hämta medelvärde och standardavvikelse från GP:n
    μ, σ = predict_y(gp, X_test_slice)
    
    # Beräkna 95% konfidensintervall
    lower_bound = μ .- 1.96 .* σ
    upper_bound = μ .+ 1.96 .* σ

    # --- PLOTTA GP (Medelvärde + Osäkerhet) ---
    plot!(p_combined, x_plot_grid, μ, 
          ribbon=(μ .- lower_bound, upper_bound .- μ), 
          fillalpha=0.2, 
          label="GP (z=$(z_val))",
          title="z=$(z_val)",
          xlabel="Kontinuerlig variabel (x)",
          ylabel="Output",
          subplot=idx,
          color=idx+1,
          lw=2
    )
    
    # --- NYTT: PLOTTA SANN FUNKTION ---
    # Vi beräknar den sanna funktionen för varje x i gridden, med fixerat z
    y_true = [true_function([x, z_val]) for x in x_plot_grid]
    
    plot!(p_combined, x_plot_grid, y_true, 
          label="Sann Funktion", 
          linestyle=:dash, # Streckad linje för sanningen
          color=:black, 
          lw=2,
          alpha=0.7,
          subplot=idx
    )

    # --- PLOTTA DATA (om det finns) ---
    z_train_rounded = round.(X_train[2, :])
    slice_indices = findall(x -> x == z_val, z_train_rounded)
    
    if !isempty(slice_indices)
        scatter!(p_combined, X_train[1, slice_indices], y_train[slice_indices],
                 label="Data",
                 markercolor=:red,
                 markersize=4,
                 subplot=idx)
    end
end

display(p_combined)

###########################################
# ==============================================================================
# 2. PLOTTA TRAPPSTEGEN (Betingat på kontinuerliga X)
# ==============================================================================

# Vi varierar Z kontinuerligt för att se stegen tydligt
z_plot_grid = range(0.4, 5.6, length=500)

# Vi väljer 4 olika värden för den KONTINUERLIGA variabeln (x)
x_fixed_values = [0.1, 0.7, 2.7, 3.9]

# Layout 2x2
plot_layout = @layout [a b; c d]
p_steps = plot(layout=plot_layout, size=(1000, 800), legend=:topleft)

for (idx, x_val) in enumerate(x_fixed_values)
    
    # Skapa input-matris (2 x N_punkter)
    # Rad 1: Fixerat x-värde
    # Rad 2: Varierande z-värde (kontinuerligt grid)
    X_test_slice = zeros(2, length(z_plot_grid))
    X_test_slice[1, :] .= x_val
    X_test_slice[2, :] .= z_plot_grid

    # Predicera
    μ, σ² = predict_f(gp, X_test_slice)
    
    # Konfidensintervall
    lower = μ .- 1.96 .* sqrt.(σ²)
    upper = μ .+ 1.96 .* sqrt.(σ²)

    # Plotta GP (Trappstegen)
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

    # Plotta "Sanningen" (för att se hur bra stegen matchar)
    # Vi beräknar den sanna funktionen för varje z i gridden
    y_true = [true_function([x_val, z]) for z in z_plot_grid]
    plot!(p_steps, z_plot_grid, y_true, 
          label="Sann funktion", 
          linestyle=:dash, 
          color=:black, 
          alpha=0.6,
          subplot=idx
    )

    # Lägg till träningsdata som råkar ligga nära detta x-värde
    # Vi visar punkter där x är inom +/- 0.25 från vårt fixerade värde
    mask = abs.(X_train[1, :] .- x_val) .< 0.2
    
    if any(mask)
        scatter!(p_steps, X_train[2, mask], y_train[mask],
                 label="Data (nära x=$(x_val))",
                 markercolor=:red,
                 markersize=5,
                 subplot=idx
        )
    end
end

display(p_steps)




startLogℓ = [0.5, 0.20]
base_k = Matern(5/2, startLogℓ, 2.0)

# Define our kernel for mixed continuous discrete, we give it the base kernel and tell it that the second input is discrete.
gm_kernel = BOOP.GarridoMerchanKernel(base_k, [2], [1:16])

gpGM = GP(X_train, y_train, MeanZero(), gm_kernel);

# --- Set priors for GP log-parameters ---
# Note that it is the log of the scales!
# Prior for continuous
priorContinuous_ℓ = Normal(log(.5), 1.5)
# Prior for discrete variable.
# Note that it is not scaled so some care should be given.
priorDisc_ℓ = Normal(log(1.5), 1.5)
# Prior for signal variance
priorProcess_σ = Normal(1.0, 3.)
# Prior for measurement noise
priorNoise_σ = Normal(log(0.1), 3.0)
# Apply the prior to the GP object
set_priors!(gpGM.logNoise, [priorNoise_σ])
set_priors!(gpGM.kernel, [priorContinuous_ℓ, priorDisc_ℓ, priorProcess_σ])

# Define some search boundaries for the optimizer, we don't want it to go completely crazy.
# Kernel lower bounds (log-scale)
lb_kern = log.([0.01, 0.5, 0.1]) 
# Kernel upper bounds (log-scale)
ub_kern = log.([10.0, 10.0, 10.0])
kern_bounds = [lb_kern, ub_kern]
# Same for noise bounds
lb_noise = log.([0.1])
ub_noise = log.([5.0])  # Max brus
noise_bounds = [lb_noise, ub_noise]
println("Training GP ...")
optOptions = Optim.Options(time_limit = 1.0, show_trace = true)




@time optimize!(gpGM, kernbounds=kern_bounds, noisebounds=noise_bounds, options=optOptions);


BOOP.propose_next(gpGM, 15.; n_restarts=10, acq_config=EIConfig(ξ=0.05))

lo = [0,1];
hi = [5,7];
baseReg_k = Matern(5/2, startLogℓ, 2.0)

gpReg = GP(rescale(X_train', lo, hi)', y_train, MeanZero(), baseReg_k);
@time optimize!(gpReg, kernbounds=kern_bounds, noisebounds=noise_bounds, options=optOptions);


pn = propose_nextt(gpReg, 15.; n_restarts=10, acq_config=EIConfig(ξ=0.05))
BOOP.inv_rescale(pn',lo,hi)  




BOOP.posteriorMax(gpGM; n_starts=20)
BOOP.posteriorMax(gpReg; n_starts=20)


plot(heatmap(gpGM, title="Garrido-Merchán"),
heatmap(gpReg, title="Regular Matérn"), size=(1400,400))
