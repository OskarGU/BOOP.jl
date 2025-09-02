

###################
using StatsBase

"""
    estimate_integral_wsabi(gp, bounds; n_samples=100_000, y_mean=0.0, y_std=1.0)

Estimates the integral of the original function f(x) using the final GP model
trained on warped data g(x) = log(f(x)).

It uses Monte Carlo integration on the posterior expectation of f(x).
E[f(x)] = exp(μ_g(x) + σ²_g(x)/2), where μ_g and σ²_g are the posterior mean
and variance of the GP fitted to the log-transformed data.

# Arguments
- `gp`: The final trained GP model.
- `bounds`: A tuple (lo, hi) defining the integration domain.
- `n_samples`: Number of Monte Carlo samples for the approximation.
- `y_mean`, `y_std`: The mean and std dev used to standardize the warped y-values,
                     needed to un-scale the GP's predictions.

# Returns
- `Float64`: The estimated value of the integral.
"""
function estimate_integral_wsabi(gp, bounds; n_samples=100_000, y_mean=0.0, y_std=1.0)
    lo, hi = bounds
    d = gp.dim

    # Calculate the volume of the integration domain
    domain_volume = prod(hi .- lo)

    # Generate a large number of random points within the original domain
    X_mc_orig = rand(d, n_samples) .* (hi .- lo) .+ lo

    # Rescale points to [-1, 1] for the GP
    X_mc_scaled = rescale(X_mc_orig', lo, hi)'

    # Get posterior mean and variance from the GP on the scaled points
    μ_scaled, σ²_scaled = predict_f(gp, X_mc_scaled)

    # --- Un-standardize the GP's predictions ---
    # The GP was trained on z = (y_warped - μ_y) / σ_y
    # So, y_warped = z * σ_y + μ_y
    μ_unstandardized = μ_scaled .* y_std .+ y_mean
    σ²_unstandardized = σ²_scaled .* (y_std^2)

    # Calculate the expected value of the original (un-warped) function f(x)
    # E[f(x)] = exp(μ_g + σ²_g/2)
    integrand_values = exp.(μ_unstandardized .+ 0.5 .* σ²_unstandardized)

    # The Monte Carlo estimate of the integral is the mean of these values times the domain volume
    integral_estimate = mean(integrand_values) * domain_volume

    return integral_estimate
end


########################
# Main WSABI function
function BQ_WSABI(f, modelSettings, optimizationSettings, warmStart)
    X, y = warmStart
    Xscaled = rescale(X, modelSettings.xBounds[1], modelSettings.xBounds[2])

    for i in 1:optimizationSettings.nIter
        # --- WARPING: Transform y-values using log ---
        # Add a small jitter to prevent log(0) for non-negative functions
        y_warped = log.(y .+ 1e-8)

        # --- Standardize the WARPED y-values ---
        μ_yw = mean(y_warped)
        σ_yw = max(std(y_warped), 1e-6) # Jitter for stability
        y_warped_scaled = (y_warped .- μ_yw) ./ σ_yw

        # Train the GP on the SCALED, WARPED data
        gp = GP(Xscaled', y_warped_scaled, modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
        optimize!(gp; kernbounds=modelSettings.kernelBounds, noisebounds=modelSettings.noiseBounds, options=Optim.Options(iterations=100))

        # Propose the next point by maximizing the posterior variance
        # The 'f_max' argument is not used by posterior_variance, so we can pass a dummy value (e.g., 0.0)
        #x_next_scaled = propose_next(gp, 0.0;
        #                             n_restarts=optimizationSettings.n_restarts,
        #                             acq=posterior_variance, dmp=nothing) # <-- Use the new acquisition function

        # Nytt anrop som skickar med stats:
        x_next_scaled = propose_nextBQ(gp, 0.0;
                             n_restarts=optimizationSettings.n_restarts,
                             acq=optimizationSettings.acq,
                             dmp=optimizationSettings.dmp,
                             stats=(μ_yw, σ_yw)) 

        # Rescale back to original bounds to evaluate the true function
        x_next_original = inv_rescale(x_next_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]

        # Evaluate the true (original) function
        y_next = 0.0
        if modelSettings.xdim == 1
            y_next = f(x_next_original[1])
        else
            y_next = f(x_next_original)
        end

        # Add the new ORIGINAL data point to the dataset
        X = vcat(X, x_next_original')
        Xscaled = vcat(Xscaled, x_next_scaled')
        y = vcat(y, y_next)

        println("Iter $i: x = $(round.(x_next_original, digits=3)), y = $(round(y_next, digits=3))")
    end

    # --- Final GP model and Integral Estimation ---
    # Warp and standardize the final dataset
    y_warped_final = log.(y .+ 1e-8)
    μ_yw_final = mean(y_warped_final)
    σ_yw_final = max(std(y_warped_final), 1e-6)
    y_ws_final = (y_warped_final .- μ_yw_final) ./ σ_yw_final

    # Fit the final, more accurate GP
    gp_final = GP(Xscaled', y_ws_final, modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
    optimize!(gp_final; kernbounds=modelSettings.kernelBounds, noisebounds=modelSettings.noiseBounds, options=Optim.Options(iterations=500))

    # Estimate the final integral using the Monte Carlo helper
    integral_estimate = estimate_integral_wsabi(gp_final, modelSettings.xBounds,
                                               n_samples=100_000,
                                               y_mean=μ_yw_final,
                                               y_std=σ_yw_final)

    println("\nFinal Integral Estimate: $integral_estimate")

    return gp_final, X, y, integral_estimate, (μ_yw_final, σ_yw_final)
end

##############################
# using GaussianProcesses, Optim, Distributions, etc.

# --- Define the function to integrate ---
# Example: A 2D Gaussian function (integral is known to be ≈ 1.0)
f(x) = exp(-20 * ( (x[1]-0.25)^2 + (x[2]-0.75)^2) ) / sqrt(2*π*0.025)

# --- 1. Model Settings ---
modelSettings_bq = (
    xdim = 2,
    xBounds = ([0.0, 0.0], [1.0, 1.0]), # Integration domain
    mean = MeanZero(),
    kernel = SE(0.0, 0.0),
    logNoise = -4.0,
    kernelBounds = [[-3., -3], [3., 3.]],
    noiseBounds = [-6., -2.]
)

# --- 2. Optimization (Sampling) Settings ---
# Naming is kept for consistency with your BO code
optimizationSettings_bq = (
    nIter = 20,
    n_restarts = 10,
    acq = posterior_variance, # <-- Key change for WSABI
    dmp=nothing
)

# --- 3. Warm Start (initial points) ---
X_start = rand(5, 2)
y_start = [f(X_start[i, :]) for i in 1:size(X_start, 1)]
warmStart_bq = (X_start, y_start)

# --- Run the WSABI algorithm ---
final_gp, X_hist, y_hist, integral_val = BQ_WSABI(f, modelSettings_bq, optimizationSettings_bq, warmStart_bq)

println("\nTrue integral is approximately 1.0")
println("WSABI estimate after $(optimizationSettings_bq.nIter) iterations: $(round(integral_val, digits=5))")

plot(final_gp)


###############
# 1d ex:
using QuadGK # Används för att beräkna den sanna integralen för jämförelse

# --- 1. Definiera en 1D-funktion att integrera ---
# En funktion med två "pucklar" på domänen [0, 6]
f(x) = exp(-(x - 2)^2) + 0.8 * exp(-(x - 4)^2 * 2)

# Beräkna den sanna integralen med hög precision för att ha ett facit
integral_true, _ = quadgk(f, 0, 6)

# --- 2. Modellanpassningar för 1D ---
modelSettings_bq_1d = (
    xdim = 1,
    xBounds = ([0.0], [6.0]), # Integrationsdomän
    mean = MeanZero(),
    kernel = SE(0.0, 0.0),
    logNoise = -4.0,
    # Bounds för kernel-hyperparametrar (log-längdskala, log-signal-std)
    kernelBounds = [[-2.0, -2.0], [2.0, 2.0]],
    noiseBounds = [-6., -2.]
)

# --- 3. Inställningar för sampling ---
optimizationSettings_bq_1d = (
    nIter = 10, # Färre iterationer behövs för att se resultat i 1D
    n_restarts = 10,
    acq = posterior_variance, # Använder osäkerhetssampling
    dmp = nothing
)

# --- 4. Startpunkter ("Warm Start") ---
# Börja med 3 slumpmässiga punkter i domänen
X_start = rand(3, 1) .* 6.0
y_start = f.(X_start)[:] # Notera den kortare syntaxen f.(X) för elementvis operation

warmStart_bq_1d = (X_start, y_start)


# --- Kör WSABI-algoritmen ---
# Se till att du har kört om definitionen av BQ_WSABI och propose_next i din session
# --- KÖR WSABI-ALGORITMEN (med uppdaterad mottagare) ---
# Vi lägger till `final_stats` för att ta emot den nya informationen
final_gp, X_hist, y_hist, integral_val, final_stats = BQ_WSABI(f, modelSettings_bq_1d, optimizationSettings_bq_1d, warmStart_bq_1d)
μ_final, σ_final = final_stats

println("\n" * "="^40)
println("Resultat:")
println("Sann integral: \t\t $(round(integral_true, digits=5))")
println("WSABI-estimat: \t $(round(integral_val, digits=5))")
println("="^40)


# --- PLOTTA RESULTATET MED JÄMFÖRELSE (KORRIGERAD) ---

# 1. Skapa ett tätt grid på den URSPRUNGLIGA skalan [0, 6] för att plotta
x_grid_original = range(modelSettings_bq_1d.xBounds[1][1], modelSettings_bq_1d.xBounds[2][1], length=400)

# 2. **NYTT STEG: Skala om gridet till [-1, 1] för prediktion**
#    (Vi måste använda reshape eftersom din rescale-funktion förväntar sig en matris)
x_grid_scaled = rescale(reshape(collect(x_grid_original), :, 1),
                        modelSettings_bq_1d.xBounds[1],
                        modelSettings_bq_1d.xBounds[2])

# 3. Hämta GP-prediktionen med det KORREKT SKALADE gridet
μ_gp_scaled, σ²_gp_scaled = predict_f(final_gp, x_grid_scaled') # Notera: x_grid_scaled'
σ_gp_scaled = sqrt.(σ²_gp_scaled)

# 4. Beräkna den sanna funktionen (samma som förut)
y_true = f.(x_grid_original)
y_true_warped = log.(y_true .+ 1e-8)
y_true_warped_scaled = (y_true_warped .- μ_final) ./ σ_final

# 5. Skapa den korrekta jämförande plotten
#    Notera att vi plottar mot `x_grid_original` på x-axeln
plot(x_grid_original, μ_gp_scaled,
     ribbon = 1.96 * σ_gp_scaled,
     fillalpha = 0.2,
     label = "GP-modell av log(f(x))",
     xlabel = "x",
     ylabel = "log(f(x)) [standardiserad]",
     title = "Jämförelse av GP-modell och Sann Funktion (Korrekt Skala)",
     legend = :bottomleft,
     linewidth = 2
)

# Lägg till den sanna funktionen
plot!(x_grid_original, y_true_warped_scaled,
    label = "Sann log(f(x))",
    color = :black,
    linestyle = :dash,
    linewidth = 2
)

# Lägg till de samplade punkterna
y_warped_scaled = (log.(y_hist .+ 1e-8) .- μ_final) ./ σ_final
scatter!(X_hist, y_warped_scaled,
         label = "Samplade punkter",
         markersize = 5,
         markerstrokewidth = 1.5,
         color = :red
)

################
# Av-warpa   
# --- PLOTTA I ORIGINALSKALAN (f(x)) ---

# Vi har redan alla värden vi behöver från föregående steg:
# μ_gp_scaled, σ_gp_scaled, μ_final, σ_final, x_grid_original

# 1. "Av-standardisera" medelvärdet och konfidensgränserna
μ_gp_warped = μ_gp_scaled .* σ_final .+ μ_final
upper_bound_warped = (μ_gp_scaled .+ 1.96 .* σ_gp_scaled) .* σ_final .+ μ_final
lower_bound_warped = (μ_gp_scaled .- 1.96 .* σ_gp_scaled) .* σ_final .+ μ_final

# 2. "Av-warpa" med exp() för att komma till originalskalan
μ_gp_original = exp.(μ_gp_warped)
upper_bound_original = exp.(upper_bound_warped)
lower_bound_original = exp.(lower_bound_warped)

# 3. Skapa plotten i originalskalan
plot(x_grid_original, μ_gp_original,
     # Ribbon hanterar asymmetriska intervall genom att ange avstånd från medelvärdet
     ribbon = (μ_gp_original .- lower_bound_original, upper_bound_original .- μ_gp_original),
     fillalpha = 0.25,
     label = "GP-modell av f(x)",
     xlabel = "x",
     ylabel = "f(x) [Originalskala]",
     title = "Slutgiltig Modell i Originalskalan",
     #legend = :topleft,
     linewidth = 2
)

# Lägg till den sanna, ursprungliga funktionen f(x)
plot!(x_grid_original, f.(x_grid_original),
    label = "Sann f(x)",
    color = :black,
    linestyle = :dash,
    linewidth = 2
)

# Lägg till de samplade punkterna (X_hist och y_hist är redan i originalskalan)
scatter!(X_hist, y_hist,
         label = "Samplade punkter",
         markersize = 5,
         markerstrokewidth = 1.5,
         color = :red
)



##################
# New ac
# Symbol för att identifiera den nya metoden
function width_in_original_scale end

"""
    original_scale_uncertainty(gp, x; y_mean, y_std, c=1.96)

Beräknar bredden på konfidensintervallet i den ursprungliga, icke-transformerade skalan.
Detta är en acquisition-funktion som balanserar exploration (osäkerhet) med
exploitation (regioner med högt förväntat funktionsvärde).

# Arguments
- `gp`: Den tränade GP-modellen (på standardiserad log-data).
- `x`: Punkten där funktionen ska utvärderas.
- `y_mean`, `y_std`: Medelvärde och std.avv. för den log-transformerade datan.
- `c`: Konstant för konfidensintervallet (1.96 för 95%).
"""
function original_scale_uncertainty(gp, x; y_mean, y_std, c=1.96)
    xvec = x isa Number ? [x] : x
    xvec = reshape(xvec, :, 1)

    # 1. Hämta prediktion från GP:n i den standardiserade log-skalan
    μ_scaled, σ²_scaled = predict_f(gp, xvec)
    μ_s, σ_s = μ_scaled[1], sqrt(max(σ²_scaled[1], 1e-8))

    # 2. "Av-standardisera" gränserna för att komma till log-skalan
    upper_bound_warped = (μ_s + c * σ_s) * y_std + y_mean
    lower_bound_warped = (μ_s - c * σ_s) * y_std + y_mean

    # 3. "Av-warpa" med exp() för att komma till originalskalan
    upper_bound_original = exp(upper_bound_warped)
    lower_bound_original = exp(lower_bound_warped)

    # 4. Returnera bredden, vilket är vår acquisition score
    return upper_bound_original - lower_bound_original
end




# Uppdatera signaturen för att kunna ta emot "stats"
function propose_nextBQ(gp, f_max; n_restarts=20, acq=expected_improvement, tuningPar = 0.10, nq=20, dmp, stats=nothing)
    d = gp.dim
    best_acq_val = -Inf
    best_x = zeros(d)

    function objective_to_minimize(x)
        val = 0.0
        if acq == expected_improvement
            val = acq(gp, x, f_max; ξ=tuningPar)
        elseif acq == upper_confidence_bound
            val = acq(gp, x; κ=tuningPar)
        elseif acq == knowledgeGradientHybrid
            val = acq(gp, x; n_z = nq)
        elseif acq == knowledgeGradientDiscrete
            val = acq(gp, x, dmp)
        elseif acq == posterior_variance
            val = acq(gp, x)
        # --- NYTT BLOCK FÖR DIN AVANCERADE FUNKTION ---
        elseif acq == width_in_original_scale
            if stats === nothing
                error("width_in_original_scale requires 'stats' (mean, std) to be provided.")
            end
            # Anropa hjälpfunktionen med den extra informationen
            val = original_scale_uncertainty(gp, x; y_mean=stats[1], y_std=stats[2])
        # ---------------------------------------------
        else
            error("Unknown acquisition function: $acq")
        end
        return -val
    end

    # ... resten av funktionen är exakt densamma ...
    for _ in 1:n_restarts
        x0 = rand(Uniform(-1., 1.), d)
        
        # ... (ingen ändring i optimeringsanropen) ...
        if acq == knowledgeGradientHybrid || acq == knowledgeGradientDiscrete
            res = optimize(objective_to_minimize, -1, 1, x0, Fminbox(NelderMead()))
        else
            res = optimize(objective_to_minimize, -1. * ones(d), 1. * ones(d), x0, Fminbox(LBFGS()); autodiff = :forward)
        end
        
        current_acq_val = -res.minimum
        if current_acq_val > best_acq_val
            best_acq_val = current_acq_val
            best_x = res.minimizer
        end
    end

    return best_x
end

# Inställningar för den nya, avancerade acquisition-funktionen
optimizationSettings_advanced = (
    nIter = 10, # Kör gärna lite fler iterationer
    n_restarts = 15,
    acq = width_in_original_scale, # <-- Välj den nya metoden
    dmp = nothing
)



# Börja med 3 slumpmässiga punkter i domänen
X_start = rand(3, 1) .* 6.0
y_start = f.(X_start)[:] # Notera den kortare syntaxen f.(X) för elementvis operation

warmStart_bq_1d = (X_start, y_start)
# Kör sedan som vanligt med de nya inställningarna
final_gp, X_hist, y_hist, integral_val, final_stats = BQ_WSABI(f, modelSettings_bq_1d, optimizationSettings_advanced, warmStart_bq_1d)

# Plotta resultaten som förut för att se skillnaden!
# ... (din plot-kod) ...
μ_final, σ_final = final_stats

println("\n" * "="^40)
println("Resultat:")
println("Sann integral: \t\t $(round(integral_true, digits=5))")
println("WSABI-estimat: \t $(round(integral_val, digits=5))")
println("="^40)


# --- PLOTTA RESULTATET MED JÄMFÖRELSE (KORRIGERAD) ---

# 1. Skapa ett tätt grid på den URSPRUNGLIGA skalan [0, 6] för att plotta
x_grid_original = range(modelSettings_bq_1d.xBounds[1][1], modelSettings_bq_1d.xBounds[2][1], length=400)

# 2. **NYTT STEG: Skala om gridet till [-1, 1] för prediktion**
#    (Vi måste använda reshape eftersom din rescale-funktion förväntar sig en matris)
x_grid_scaled = rescale(reshape(collect(x_grid_original), :, 1),
                        modelSettings_bq_1d.xBounds[1],
                        modelSettings_bq_1d.xBounds[2])

# 3. Hämta GP-prediktionen med det KORREKT SKALADE gridet
μ_gp_scaled, σ²_gp_scaled = predict_f(final_gp, x_grid_scaled') # Notera: x_grid_scaled'
σ_gp_scaled = sqrt.(σ²_gp_scaled)

# 4. Beräkna den sanna funktionen (samma som förut)
y_true = f.(x_grid_original)
y_true_warped = log.(y_true .+ 1e-8)
y_true_warped_scaled = (y_true_warped .- μ_final) ./ σ_final

# 5. Skapa den korrekta jämförande plotten
#    Notera att vi plottar mot `x_grid_original` på x-axeln
plot(x_grid_original, μ_gp_scaled,
     ribbon = 1.96 * σ_gp_scaled,
     fillalpha = 0.2,
     label = "GP-modell av log(f(x))",
     xlabel = "x",
     ylabel = "log(f(x)) [standardiserad]",
     title = "Jämförelse av GP-modell och Sann Funktion (Korrekt Skala)",
     legend = :bottomleft,
     linewidth = 2
)

# Lägg till den sanna funktionen
plot!(x_grid_original, y_true_warped_scaled,
    label = "Sann log(f(x))",
    color = :black,
    linestyle = :dash,
    linewidth = 2
)

# Lägg till de samplade punkterna
y_warped_scaled = (log.(y_hist .+ 1e-8) .- μ_final) ./ σ_final
scatter!(X_hist, y_warped_scaled,
         label = "Samplade punkter",
         markersize = 5,
         markerstrokewidth = 1.5,
         color = :red
)

################
# Av-warpa   
# --- PLOTTA I ORIGINALSKALAN (f(x)) ---

# Vi har redan alla värden vi behöver från föregående steg:
# μ_gp_scaled, σ_gp_scaled, μ_final, σ_final, x_grid_original

# 1. "Av-standardisera" medelvärdet och konfidensgränserna
μ_gp_warped = μ_gp_scaled .* σ_final .+ μ_final
upper_bound_warped = (μ_gp_scaled .+ 1.96 .* σ_gp_scaled) .* σ_final .+ μ_final
lower_bound_warped = (μ_gp_scaled .- 1.96 .* σ_gp_scaled) .* σ_final .+ μ_final

# 2. "Av-warpa" med exp() för att komma till originalskalan
μ_gp_original = exp.(μ_gp_warped)
upper_bound_original = exp.(upper_bound_warped)
lower_bound_original = exp.(lower_bound_warped)

# 3. Skapa plotten i originalskalan
plot(x_grid_original, μ_gp_original,
     # Ribbon hanterar asymmetriska intervall genom att ange avstånd från medelvärdet
     ribbon = (μ_gp_original .- lower_bound_original, upper_bound_original .- μ_gp_original),
     fillalpha = 0.25,
     label = "GP-modell av f(x)",
     xlabel = "x",
     ylabel = "f(x) [Originalskala]",
     title = "Slutgiltig Modell i Originalskalan",
     #legend = :topleft,
     linewidth = 2
)

# Lägg till den sanna, ursprungliga funktionen f(x)
plot!(x_grid_original, f.(x_grid_original),
    label = "Sann f(x)",
    color = :black,
    linestyle = :dash,
    linewidth = 2
)

# Lägg till de samplade punkterna (X_hist och y_hist är redan i originalskalan)
scatter!(X_hist, y_hist,
         label = "Samplade punkter",
         markersize = 5,
         markerstrokewidth = 1.5,
         color = :red
)







#####################
#####################
# med tuning c.
# Ladda alla nödvändiga paket
using GaussianProcesses
using Optim
using Distributions
using Plots
using QuadGK

# --- Funktioner (från tidigare) ---

# Samma funktion att integrera
f(x) = log.((exp(-(x - 2)^2) + 0.8 * exp(-(x - 4)^2 * 2) )*exp(randn()*0.05))

fNF(x) = log.(exp(-(x - 2)^2) + 0.8 * exp(-(x - 4)^2 * 2))

integral_true, _ = quadgk(fNF, 0, 6)

# Symbol och hjälpfunktion för vår avancerade acquisition-funktion
function width_in_original_scale end
function original_scale_uncertainty(gp, x; y_mean, y_std, c=1.96)
    xvec = x isa Number ? [x] : x
    xvec = reshape(xvec, :, 1)
    μ_scaled, σ²_scaled = predict_f(gp, xvec)
    μ_s, σ_s = μ_scaled[1], sqrt(max(σ²_scaled[1], 1e-8))
    upper_bound_warped = (μ_s + c * σ_s) * y_std + y_mean
    lower_bound_warped = (μ_s - c * σ_s) * y_std + y_mean
    upper_bound_original = exp(upper_bound_warped)
    lower_bound_original = exp(lower_bound_warped)
    return upper_bound_original - lower_bound_original
end

# Andra hjälpfunktioner (rescale, etc. antas finnas definierade)
function rescale2(X, lo, hi)
    return 2 .* (X .- lo') ./ (hi' .- lo') .- 1
end
inv_rescale(X_scaled, lo, hi) = ((X_scaled .+ 1) ./ 2) .* (hi' .- lo') .+ lo'

# -------------------------------------------------------------
# --- STEG 1: Uppdatera `propose_nextBQ` ---
# -------------------------------------------------------------
function propose_nextBQ(gp, f_max; n_restarts=20, acq=expected_improvement, tuningPar = 0.10, nq=20, dmp, stats=nothing)
    d = gp.dim
    best_acq_val = -Inf
    best_x = zeros(d)

    function objective_to_minimize(x)
        val = 0.0
        # --- (Inga ändringar i de första blocken) ---
        if acq == posterior_variance
             val = posterior_variance(gp, x)
        # --- ÄNDRING I DETTA BLOCK ---
        elseif acq == width_in_original_scale
            if stats === nothing || length(stats) < 3
                error("width_in_original_scale requires 'stats' (mean, std, c) to be provided.")
            end
            # Anropa hjälpfunktionen med den dynamiska c-parametern från stats[3]
            val = original_scale_uncertainty(gp, x; y_mean=stats[1], y_std=stats[2], c=stats[3]) # <-- ÄNDRING
        # ---------------------------------------------
        else
            error("Unknown acquisition function: $acq")
        end
        return -val
    end

    for _ in 1:n_restarts
        x0 = rand(Uniform(-1., 1.), d)
        if acq == knowledgeGradientHybrid || acq == knowledgeGradientDiscrete
            res = optimize(objective_to_minimize, -1, 1, x0, Fminbox(NelderMead()))
        else
            res = optimize(objective_to_minimize, -1. * ones(d), 1. * ones(d), x0, Fminbox(LBFGS()); autodiff = :forward)
        end
        current_acq_val = -res.minimum
        if current_acq_val > best_acq_val
            best_acq_val = current_acq_val
            best_x = res.minimizer
        end
    end

    return best_x
end

# -------------------------------------------------------------
# --- STEG 2: Uppdatera `BQ_WSABI` ---
# -------------------------------------------------------------
function BQ_WSABI(f, modelSettings, optimizationSettings, warmStart)
    X, y = warmStart
    Xscaled = rescale(X, modelSettings.xBounds[1], modelSettings.xBounds[2])

    for i in 1:optimizationSettings.nIter
        y_warped = y #log.(y .+ 1e-8)
        μ_yw = mean(y_warped)
        σ_yw = max(std(y_warped), 1e-6)
        y_warped_scaled = (y_warped .- μ_yw) ./ σ_yw

        gp = GP(Xscaled', y_warped_scaled, modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
        optimize!(gp; kernbounds=modelSettings.kernelBounds, noisebounds=modelSettings.noiseBounds, options=Optim.Options(iterations=100))

        # --- NYTT BLOCK: Beräkna dynamisk c-parameter ---
        # Hämta posterior-medelvärdet vid de redan observerade punkterna
        μ_obs_scaled, _ = predict_f(gp, Xscaled')
        # "Av-standardisera" för att få dem på log-skalan
        μ_obs_warped = vec(μ_obs_scaled) .* σ_yw .+ μ_yw
        # Beräkna rangen (max - min)
        dr_posterior = maximum(μ_obs_warped) - minimum(μ_obs_warped)

        # Beräkna dynamiskt c med den logaritmiska heuristiken
        c_base = optimizationSettings.c_base
        c_sensitivity = optimizationSettings.c_sensitivity
        c_dynamisk = c_base + c_sensitivity * log10(1 + dr_posterior)
        # ---------------------------------------------------

        # Anropa propose_nextBQ och skicka med den nya dynamiska c-parametern
        x_next_scaled = propose_nextBQ(gp, 0.0;
                                     n_restarts=optimizationSettings.n_restarts,
                                     acq=optimizationSettings.acq,
                                     dmp=optimizationSettings.dmp,
                                     stats=(μ_yw, σ_yw, c_dynamisk)) # <-- ÄNDRING

        x_next_original = inv_rescale(x_next_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]
        y_next = f(x_next_original[1])

        X = vcat(X, x_next_original')
        Xscaled = vcat(Xscaled, x_next_scaled')
        y = vcat(y, y_next)

        println("Iter $i: Dynamic Range (post-mean) = $(round(dr_posterior, digits=2)), Använder c = $(round(c_dynamisk, digits=2))")
    end

    # ... (resten av funktionen för slutgiltig GP och integral är oförändrad) ...
    y_warped_final = y#log.(y .+ 1e-8)
    μ_yw_final = mean(y_warped_final)
    σ_yw_final = max(std(y_warped_final), 1e-6)
    y_ws_final = (y_warped_final .- μ_yw_final) ./ σ_yw_final

    gp_final = GP(Xscaled', y_ws_final, modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
    optimize!(gp_final; kernbounds=modelSettings.kernelBounds, noisebounds=modelSettings.noiseBounds, options=Optim.Options(iterations=500))

    integral_estimate = estimate_integral_wsabi(gp_final, modelSettings.xBounds, n_samples=100_000, y_mean=μ_yw_final, y_std=σ_yw_final)
    
    return gp_final, X, y, integral_estimate, (μ_yw_final, σ_yw_final)
end

# Funktion för att estimera integralen (oförändrad)
function estimate_integral_wsabi(gp, bounds; n_samples=100_000, y_mean=0.0, y_std=1.0)
    lo, hi = bounds
    d = gp.dim
    domain_volume = prod(hi .- lo)
    X_mc_orig = rand(d, n_samples) .* (hi .- lo) .+ lo
    X_mc_scaled = rescale(X_mc_orig', lo, hi)'
    μ_scaled, σ²_scaled = predict_f(gp, X_mc_scaled)
    μ_unstandardized = μ_scaled .* y_std .+ y_mean
    σ²_unstandardized = σ²_scaled .* (y_std^2)
    integrand_values = exp.(μ_unstandardized .+ 0.5 .* σ²_unstandardized)
    integral_estimate = mean(integrand_values) * domain_volume
    return integral_estimate
end


# -------------------------------------------------------------
# --- STEG 3: Kör med nya adaptiva inställningar ---
# -------------------------------------------------------------

modelSettings_bq_1d = (
    xdim = 1,
    xBounds = ([0.0], [6.0]),
    mean = MeanZero(),
    kernel = SE(0.0, 0.0),
    logNoise = -4.0,
    kernelBounds = [[-2.0, -2.0], [2.0, 2.0]],
    noiseBounds = [-6., -2.]
)

# --- NYTT: Inställningar för den ADAPTIVA acquisition-funktionen ---
optimizationSettings_adaptive = (
    nIter = 16,
    n_restarts = 15,
    acq = width_in_original_scale,
    dmp = nothing,
    c_base = 1.96,        # <-- Grundläggande "äventyrlighet"
    c_sensitivity = 0.5   # <-- Hur starkt c ska reagera på dynamisk range
)

# Startpunkter
X_start = rand(3, 1) .* 6.0
y_start = vec(f.(X_start))
warmStart_bq_1d = (X_start, y_start)

# Kör algoritmen!
final_gp, X_hist, y_hist, integral_val, final_stats = BQ_WSABI(f, modelSettings_bq_1d, optimizationSettings_adaptive, warmStart_bq_1d)

# Plotta resultaten som förut och se hur c förändras i konsolen!
# ... (all din plot-kod kan återanvändas här) ...
# Plotta resultaten som förut för att se skillnaden!
# ... (din plot-kod) ...
μ_final, σ_final = final_stats

println("\n" * "="^40)
println("Resultat:")
println("Sann integral: \t\t $(round(integral_true, digits=5))")
println("WSABI-estimat: \t $(round(integral_val, digits=5))")
println("="^40)


# --- PLOTTA RESULTATET MED JÄMFÖRELSE (KORRIGERAD) ---

# 1. Skapa ett tätt grid på den URSPRUNGLIGA skalan [0, 6] för att plotta
x_grid_original = range(modelSettings_bq_1d.xBounds[1][1], modelSettings_bq_1d.xBounds[2][1], length=400)

# 2. **NYTT STEG: Skala om gridet till [-1, 1] för prediktion**
#    (Vi måste använda reshape eftersom din rescale-funktion förväntar sig en matris)
x_grid_scaled = rescale(reshape(collect(x_grid_original), :, 1),
                        modelSettings_bq_1d.xBounds[1],
                        modelSettings_bq_1d.xBounds[2])

# 3. Hämta GP-prediktionen med det KORREKT SKALADE gridet
μ_gp_scaled, σ²_gp_scaled = predict_f(final_gp, x_grid_scaled') # Notera: x_grid_scaled'
σ_gp_scaled = sqrt.(σ²_gp_scaled)

# 4. Beräkna den sanna funktionen (samma som förut)
y_true = fNF.(x_grid_original)
y_true_warped = y_true #log.(y_true .+ 1e-8)
y_true_warped_scaled = (y_true_warped .- μ_final) ./ σ_final

# 5. Skapa den korrekta jämförande plotten
#    Notera att vi plottar mot `x_grid_original` på x-axeln
plot(x_grid_original, μ_gp_scaled,
     ribbon = 1.96 * σ_gp_scaled,
     fillalpha = 0.2,
     label = "GP-modell av log(f(x))",
     xlabel = "x",
     ylabel = "log(f(x)) [standardiserad]",
     title = "Jämförelse av GP-modell och Sann Funktion (Korrekt Skala)",
     legend = :bottomleft,
     linewidth = 2
)

# Lägg till den sanna funktionen
plot!(x_grid_original, y_true_warped_scaled,
    label = "Sann log(f(x))",
    color = :black,
    linestyle = :dash,
    linewidth = 2
)

# Lägg till de samplade punkterna
y_warped_scaled = (log.(y_hist .+ 1e-8) .- μ_final) ./ σ_final
scatter!(X_hist, y_warped_scaled,
         label = "Samplade punkter",
         markersize = 5,
         markerstrokewidth = 1.5,
         color = :red
)

################
# Av-warpa   
# --- PLOTTA I ORIGINALSKALAN (f(x)) ---

# Vi har redan alla värden vi behöver från föregående steg:
# μ_gp_scaled, σ_gp_scaled, μ_final, σ_final, x_grid_original

# 1. "Av-standardisera" medelvärdet och konfidensgränserna
μ_gp_warped = μ_gp_scaled .* σ_final .+ μ_final
upper_bound_warped = (μ_gp_scaled .+ 1.96 .* σ_gp_scaled) .* σ_final .+ μ_final
lower_bound_warped = (μ_gp_scaled .- 1.96 .* σ_gp_scaled) .* σ_final .+ μ_final

# 2. "Av-warpa" med exp() för att komma till originalskalan
μ_gp_original = exp.(μ_gp_warped)
upper_bound_original = exp.(upper_bound_warped)
lower_bound_original = exp.(lower_bound_warped)

# 3. Skapa plotten i originalskalan
plot(x_grid_original, μ_gp_original,
     # Ribbon hanterar asymmetriska intervall genom att ange avstånd från medelvärdet
     ribbon = (μ_gp_original .- lower_bound_original, upper_bound_original .- μ_gp_original),
     fillalpha = 0.25,
     label = "GP-modell av f(x)",
     xlabel = "x",
     ylabel = "f(x) [Originalskala]",
     title = "Slutgiltig Modell i Originalskalan",
     #legend = :topleft,
     linewidth = 2
)

# Lägg till den sanna, ursprungliga funktionen f(x)
plot!(x_grid_original, exp.(fNF.(x_grid_original)),
    label = "Sann f(x)",
    color = :black,
    linestyle = :dash,
    linewidth = 2
)

# Lägg till de samplade punkterna (X_hist och y_hist är redan i originalskalan)
scatter!(X_hist, exp.(y_hist),
         label = "Samplade punkter",
         markersize = 5,
         markerstrokewidth = 1.5,
         color = :red
)


