using Distributions, LinearAlgebra

"""
    kalman_filter_loglik(log_params, y_obs)

Beräknar log-likelihooden för en Local Level Model med Kalmanfiltret.
Tar logaritmerade varianser som input för att säkerställa positivitet.
"""
function kalman_filter_loglik(log_params::Vector{Float64}, y_obs::Vector{Float64})
    # Parametrar på originalskalan
    σ2_ε = exp(log_params[1]) # Observationsvarians
    σ2_η = exp(log_params[2]) # Tillståndsvarians

    T = length(y_obs)
    loglik = 0.0

    # Initialisering (diffus prior)
    a = 0.0 # a_1|0
    P = 1e7 # P_1|0

    for t in 1:T
        # Prediktionssteg
        # a_t|t-1 = a_t-1|t-1  (ingen dynamik i Local Level)
        # P_t|t-1 = P_t-1|t-1 + σ²_η
        P = P + σ2_η

        # Uppdateringssteg
        v = y_obs[t] - a       # Prediktionsfel
        F = P + σ2_ε           # Prediktionsfelets varians
        K = P / F              # Kalman Gain

        a = a + K * v          # a_t|t
        P = P - K * P          # P_t|t

        # Addera till log-likelihood
        loglik += logpdf(Normal(0, sqrt(F)), v)
    end
    return loglik
end

"""
    kalman_smoother(log_params, y_obs)

Kör Kalmanfilter och Rauch-Tung-Striebel (RTS) smoother för en Local Level Model.
Returnerar de utjämnade (smoothed) medelvärdena och varianserna för tillståndet μ_t.
"""
function kalman_smoother(log_params::Vector{Float64}, y_obs::Vector{Float64})
    # Parametrar
    σ2_ε = exp(log_params[1])
    σ2_η = exp(log_params[2])
    T = length(y_obs)

    # --- Steg 1: Kör Kalmanfiltret framåt och lagra resultat ---
    a_filtered = zeros(T)
    P_filtered = zeros(T)
    P_predicted = zeros(T) # Behövs för smoothern

    a = 0.0; P = 1e7

    for t in 1:T
        # Prediktion
        P_pred = P + σ2_η
        P_predicted[t] = P_pred

        # Uppdatering
        v = y_obs[t] - a
        F = P_pred + σ2_ε
        K = P_pred / F
        a = a + K * v
        P = P_pred - K * P_pred

        a_filtered[t] = a
        P_filtered[t] = P
    end

    # --- Steg 2: Kör RTS-smoothern bakåt ---
    a_smoothed = zeros(T); P_smoothed = zeros(T)
    a_smoothed[T] = a_filtered[T]
    P_smoothed[T] = P_filtered[T]

    for t in (T-1):-1:1
        J = P_filtered[t] / P_predicted[t+1]
        a_smoothed[t] = a_filtered[t] + J * (a_smoothed[t+1] - a_filtered[t])
        P_smoothed[t] = P_filtered[t] + J * (P_smoothed[t+1] - P_predicted[t+1]) * J
    end

    return a_smoothed, P_smoothed
end

#################################
"""
    approximate_smoothing_posterior(y_obs, priors, bo_settings; grid_points_per_dim=7)

Huvudfunktion för att approximera smoothing-posteriorn för en Local Level Model.

Använder Bayesiansk Optimering för att bygga en surrogatmodell av
hyperparametrarnas posteriorfördelning, och sedan numerisk kvadratur
för att marginalisera ut osäkerheten i hyperparametrarna.

# Argument
- `y_obs`: Vektorn med tidsseriedata.
- `priors`: En Dict med priorfördelningar för 'log_sigma2_eps' och 'log_sigma2_eta'.
- `bo_settings`: En NamedTuple med inställningar för BO-loopen.
- `grid_points_per_dim`: Antal punkter per dimension i kvadratur-rutnätet.

# Returnerar
- `smoothed_mean`: Marginaliserat medelvärde för tillståndet μ_t.
- `smoothed_var`: Marginaliserad varians för tillståndet μ_t.
- `gp_posterior`: Den tränade GP-modellen för hyperparametrarnas posterior.
- `theta_map`: MAP-skattningen för log-varianserna.
"""
function approximate_smoothing_posterior(y_obs::Vector{Float64}, priors::Dict, bo_settings::NamedTuple; grid_points_per_dim=7)

    # 1. Definiera den onormaliserade log-posteriorn som vår BO ska maximera
    function log_posterior(log_params)
        # Log-likelihood från Kalmanfiltret
        loglik = kalman_filter_loglik(log_params, y_obs)
        if !isfinite(loglik) return -Inf end # Skydd mot ogiltiga parametrar

        # Addera log-priors
        logp = logpdf(priors["log_sigma2_eps"], log_params[1]) +
               logpdf(priors["log_sigma2_eta"], log_params[2])

        return loglik + logp
    end

    # 2. Konfigurera och kör Bayesiansk Optimering för att träna GP:n
    # Vi behöver en startpunkt för BO. Vi kan dra en från priorn.
    start_x = [rand(priors["log_sigma2_eps"]); rand(priors["log_sigma2_eta"])]'
    start_y = [log_posterior(start_x')]

    gp_posterior, _, _, theta_map_vec, _, _ = BO(log_posterior, bo_settings.model, bo_settings.opt, (start_x, start_y))
    theta_map = theta_map_vec' # MAP-skattningen av log-parametrarna

    println("\nBO-fasen klar. MAP-skattning för (log_σ²_ε, log_σ²_η): $(round.(theta_map, digits=3))")

    # 3. Skapa ett kvadratur-rutnät runt MAP-skattningen
    # Vi använder GP-kernelns längdskalor som ett mått på posteriorns bredd
    # för att skapa ett anpassat rutnät.
    ls = gp_posterior.kernel.ℓ2
    grid_width_scale = 2.0 # Hur många längdskalor brett rutnätet ska vara

    dim1_range = range(theta_map[1] - grid_width_scale * ls, theta_map[1] + grid_width_scale * ls, length=grid_points_per_dim)
    dim2_range = range(theta_map[2] - grid_width_scale * ls, theta_map[2] + grid_width_scale * ls, length=grid_points_per_dim)

    grid_iterator = Iterators.product(dim1_range, dim2_range)
    theta_grid = hcat([[d1, d2] for (d1, d2) in grid_iterator]...) # 2xN matris

    # 4. Beräkna vikterna för varje punkt i rutnätet med hjälp av GP:n
    log_p_values, _ = predict_f(gp_posterior, theta_grid)
    
    # Numeriskt stabil softmax för att få vikter
    p_values = exp.(log_p_values .- maximum(log_p_values))
    weights = p_values ./ sum(p_values)
    weights = vec(weights) # Säkerställ att det är en vektor

    println("Kvadratur-fasen: Beräknar smoother för $(length(weights)) viktade punkter.")

    # 5. Kör smoothern för varje punkt och beräkna det viktade medelvärdet
    T = length(y_obs)
    final_smoothed_mean = zeros(T)
    # För variansen använder vi lagen om total varians: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
    E_Var = zeros(T) # E[P_smoothed | y]
    E_Mean_sq = zeros(T) # E[a_smoothed^2 | y]

    for i in 1:size(theta_grid, 2)
        theta_i = theta_grid[:, i]
        w_i = weights[i]

        a_hat_i, P_hat_i = kalman_smoother(theta_i, y_obs)

        final_smoothed_mean .+= w_i .* a_hat_i
        E_Var .+= w_i .* P_hat_i
        E_Mean_sq .+= w_i .* (a_hat_i.^2)
    end

    Var_E = E_Mean_sq - final_smoothed_mean.^2 # Var(E[μ|y,θ])
    final_smoothed_var = E_Var + Var_E

    return final_smoothed_mean, final_smoothed_var, gp_posterior, theta_map
end


###################################
# Ladda dina befintliga BO-paketfiler
# include("path/to/your/acq_functions.jl")
# include("path/to/your/bo_main.jl") etc.
using GaussianProcesses, Optim, Distributions, Plots

# --- 1. Generera syntetisk testdata ---
Random.seed!(123)
T = 100
σ_ε_true = 0.5
σ_η_true = 0.1
μ_true = cumsum(rand(Normal(0, σ_η_true), T))
y_obs = μ_true + rand(Normal(0, σ_ε_true), T)

plot(1:T, μ_true, label="Sann latent variabel μ_t", lw=2)
scatter!(1:T, y_obs, label="Observerad data y_t", alpha=0.6, ms=3)
title!("Syntetisk tidsseriedata")

# --- 2. Definiera priors och BO-inställningar ---
priors = Dict(
    "log_sigma2_eps" => Normal(-1.0, 2.0), # Prior för log(σ²_ε)
    "log_sigma2_eta" => Normal(-2.0, 2.0)  # Prior för log(σ²_η)
)

# Inställningar för din BO-funktion
model_settings = (
    xdim = 2,
    xBounds = ([-10., -10.], [5., 5.]), # Sökrymd för log-varianserna
    mean = MeanZero(),
    kernel = SE(0.0, 0.0),
    logNoise = -2.0,
    kernelBounds = [[-3., -3.], [3., 3.]],
    noiseBounds = [-4., 0.]
)

opt_settings = (
    nIter = 60,
    n_restarts = 10,
    acq = expected_improvement, # UCB är ofta bra för att utforska en yta
    tuningPar = 0.3, # Högre kappa för mer exploration
    nq = 0, # Inte relevant för UCB
    dmp = nothing # Inte relevant för UCB
)

bo_settings = (model = model_settings, opt = opt_settings)


# --- 3. Kör huvudfunktionen ---
smoothed_mean, smoothed_var, gp, theta_map = approximate_smoothing_posterior(y_obs, priors, bo_settings; grid_points_per_dim=7);

# --- 4. Plotta resultaten ---
smoothed_std = sqrt.(smoothed_var)
ci_upper = smoothed_mean .+ 1.96 .* smoothed_std
ci_lower = smoothed_mean .- 1.96 .* smoothed_std

plot(1:T, μ_true, label="Sann latent variabel μ_t", lw=2, color=:black)
plot!(1:T, smoothed_mean, label="Smoothed Mean E[μ_t|y]", lw=2, color=:red, ribbon=(smoothed_mean .- ci_lower, ci_upper .- smoothed_mean), fillalpha=0.2)
scatter!(1:T, y_obs, label="Observerad data y_t", alpha=0.4, ms=3)
title!("Resultat från BQ-baserad Smoothing")

# Plotta posteriorn för hyperparametrarna
grid1 = range(model_settings.xBounds[1][1], model_settings.xBounds[2][1], length=50)
grid2 = range(model_settings.xBounds[1][2], model_settings.xBounds[2][2], length=50)
log_posterior_surface, _ = predict_f(gp, hcat([[g1, g2] for g1 in grid1 for g2 in grid2]...))
posterior_surface = exp.(reshape(log_posterior_surface, 50, 50))

true_log_params = [log(σ_ε_true^2), log(σ_η_true^2)]

contourf(grid1, grid2, posterior_surface', title="Approximerad Posterior p(logσ²|y)", xlabel="log(σ²_ε)", ylabel="log(σ²_η)")
scatter!([theta_map[1]], [theta_map[2]], label="MAP-skattning", markersize=8, color=:yellow)
scatter!([true_log_params[1]], [true_log_params[2]], label="Sanna värden", markersize=8, color=:red, marker=:xcross)



#################
# Gemini gibbs:
using Distributions, LinearAlgebra, ProgressMeter

"""
    ffbs(y, a0, P0, σ2_ε, σ2_η)

Forward Filtering, Backward Sampling (FFBS) algorithm.
Drar ett sample från p(μ_1:T | y_1:T, parameters).
"""
function ffbs(y, a0, P0, σ2_ε, σ2_η)
    T = length(y)
    
    # --- Forward Filtering (Kalman Filter) ---
    a_filtered = zeros(T); P_filtered = zeros(T)
    a_pred = zeros(T); P_pred = zeros(T)
    
    a = a0; P = P0
    for t in 1:T
        # Prediction
        a_pred[t] = a
        P_pred[t] = P + σ2_η
        
        # Update
        v = y[t] - a_pred[t]
        F = P_pred[t] + σ2_ε
        K = P_pred[t] / F
        a = a_pred[t] + K * v
        P = P_pred[t] - K * P_pred[t]
        
        a_filtered[t] = a
        P_filtered[t] = P
    end

    # --- Backward Sampling ---
    μ_sample = zeros(T)
    μ_sample[T] = rand(Normal(a_filtered[T], sqrt(P_filtered[T])))

    for t in (T-1):-1:1
        # Smoother mean and variance
        J = P_filtered[t] / P_pred[t+1]
        a_smooth = a_filtered[t] + J * (μ_sample[t+1] - a_pred[t+1])
        P_smooth = P_filtered[t] - J * P_pred[t+1] * J
        
        μ_sample[t] = rand(Normal(a_smooth, sqrt(max(P_smooth, 1e-9))))
    end
    
    return μ_sample
end


"""
    gibbs_sampler_local_level(y_obs, prior_shapes, prior_scales, n_samples, n_burnin)

Kör en Gibbs-sampler för en Local Level Model.
"""
function gibbs_sampler_local_level(y_obs, prior_shapes, prior_scales, n_samples, n_burnin)
    T = length(y_obs)
    total_iter = n_samples + n_burnin

    # Prior-parametrar för Inverse Gamma: IG(shape, scale)
    alpha_eps, beta_eps = prior_shapes[1], prior_scales[1]
    alpha_eta, beta_eta = prior_shapes[2], prior_scales[2]

    # Initialvärden
    σ2_ε_sample = 1.0 / rand(Gamma(alpha_eps, 1/beta_eps))
    σ2_η_sample = 1.0 / rand(Gamma(alpha_eta, 1/beta_eta))
    μ_sample = copy(y_obs) # Starta med μ_t = y_t

    # Lagring för posterior samples
    σ2_ε_samples = zeros(n_samples)
    σ2_η_samples = zeros(n_samples)
    μ_samples = zeros(T, n_samples)
    
    @showprogress "Gibbs Sampling..." for i in 1:total_iter
        # 1. Sampla μ_1:T | y, σ²_ε, σ²_η
        # Vi använder FFBS. a0 och P0 är startvärden för Kalmanfiltret.
        μ_sample = ffbs(y_obs, 0.0, 1e7, σ2_ε_sample, σ2_η_sample)
        
        # 2. Sampla σ²_ε | y, μ_1:T, σ²_η
        sse_ε = sum((y_obs - μ_sample).^2)
        post_shape_ε = alpha_eps + T / 2
        post_scale_ε = beta_eps + sse_ε / 2
        σ2_ε_sample = 1.0 / rand(Gamma(post_shape_ε, 1/post_scale_ε))

        # 3. Sampla σ²_η | y, μ_1:T, σ²_ε
        sse_η = sum(diff(μ_sample).^2)
        post_shape_eta = alpha_eta + (T - 1) / 2
        post_scale_eta = beta_eta + sse_η / 2
        σ2_η_sample = 1.0 / rand(Gamma(post_shape_eta, 1/post_scale_eta))

        # Lagra samples efter burn-in
        if i > n_burnin
            idx = i - n_burnin
            σ2_ε_samples[idx] = σ2_ε_sample
            σ2_η_samples[idx] = σ2_η_sample
            μ_samples[:, idx] = μ_sample
        end
    end

    return (σ2_ε=σ2_ε_samples, σ2_η=σ2_η_samples, μ=μ_samples)
end

gibbs_mean_mu = mean(gibbs_results.μ, dims=2)
gibbs_lower_ci = [quantile(gibbs_results.μ[t, :], 0.025) for t in 1:T]
gibbs_upper_ci = [quantile(gibbs_results.μ[t, :], 0.975) for t in 1:T]



# ==============================================================================
# DEL 2: Huvudskript för Jämförelse
# ==============================================================================
using StatsPlots
# --- 1. Generera syntetisk testdata ---
Random.seed!(42)
T = 150
σ_ε_true = 0.8
σ_η_true = 0.2
μ_true = cumsum(rand(Normal(0, σ_η_true), T))
y_obs = μ_true + rand(Normal(0, σ_ε_true), T)
true_log_params = [log(σ_ε_true^2), log(σ_η_true^2)]

println("Sanna log-varians parametrar: $(round.(true_log_params, digits=3))")

# --- 2. Kör BQ/BO-baserad inferens ---
priors_bq = Dict( # Priors för log-varianser
    "log_sigma2_eps" => Normal(0, 3.0),
    "log_sigma2_eta" => Normal(-2, 3.0)
)
model_settings_bq = (
    xdim = 2, xBounds = ([-5., -5.], [2., 2.]), mean = MeanZero(),
    kernel = SE(0.0, 0.0), logNoise = -3.0,
    kernelBounds = [[-5., -5.], [5., 5.]], noiseBounds = [-6., -1.]
)
opt_settings_bq = (
    nIter = 200, n_restarts = 15, acq = expected_improvement,
    tuningPar = 0.3, nq = 0, dmp = nothing
)
bo_settings = (model = model_settings_bq, opt = opt_settings_bq)

bq_mean, bq_var, gp_posterior, theta_map_bq = approximate_smoothing_posterior(y_obs, priors_bq, bo_settings; grid_points_per_dim=9);

# --- 3. Kör Gibbs-sampler ---
# Svagt informativa prioter för varianserna (IG(0.01, 0.01))prior_shapes_gibbs = [0.01, 0.01]
prior_scales_gibbs = [0.01, 0.01]
gibbs_results = gibbs_sampler_local_level(y_obs, prior_shapes_gibbs, prior_scales_gibbs, 40000, 5000);

# --- 4. Jämför de statiska parametrarna (Plot A) ---
log_s2_eps_gibbs = log.(gibbs_results.σ2_ε)
log_s2_eta_gibbs = log.(gibbs_results.σ2_η)
gibbs_mean_params = [mean(log_s2_eps_gibbs), mean(log_s2_eta_gibbs)]

# Skapa rutnät för BQ/BO-ytan
grid1 = range(model_settings_bq.xBounds[1][1], model_settings_bq.xBounds[2][1], length=500)
grid2 = range(model_settings_bq.xBounds[1][2], model_settings_bq.xBounds[2][2], length=500)
log_posterior_surface, _ = predict_f(gp_posterior, hcat([[g1, g2] for g1 in grid1 for g2 in grid2]...))
posterior_surface_bq = exp.(reshape(log_posterior_surface, 500, 500))

# Skapa plotten
plot_params = contour(grid1, grid2, posterior_surface_bq', fill=true, c=:viridis,
    title="Jämförelse av Posterior för Hyperparametrar",
    xlabel="log(σ²_ε)", ylabel="log(σ²_η)",
    legend=:topleft)
density!(plot_params, log_s2_eps_gibbs, log_s2_eta_gibbs,
         color=:red, linewidth=2, label="Gibbs Posterior Densitet", levels=5)
scatter!(plot_params, [theta_map_bq[1]], [theta_map_bq[2]],
         label="BQ/BO MAP", markersize=8, color=:yellow, marker=:star5)
scatter!(plot_params, [gibbs_mean_params[1]], [gibbs_mean_params[2]],
        label="Gibbs Posterior Mean", markersize=6, color=:cyan, marker=:circle)
scatter!(plot_params, [true_log_params[1]], [true_log_params[2]],
         label="Sanna Värden", markersize=8, color=:magenta, marker=:xcross, ms=10, strokewidth=3)


# --- 5. Jämför de utjämnade tillstånden (Plot B) ---
# Beräkna statistik från Gibbs
gibbs_mean_mu = vec(mean(gibbs_results.μ, dims=2))
gibbs_lower_ci = [quantile(gibbs_results.μ[t, :], 0.025) for t in 1:T]
gibbs_upper_ci = [quantile(gibbs_results.μ[t, :], 0.975) for t in 1:T]

# Beräkna statistik från BQ/BO
bq_std = sqrt.(bq_var)
bq_lower_ci = bq_mean .- 1.96 .* bq_std
bq_upper_ci = bq_mean .+ 1.96 .* bq_std

# Skapa plotten
plot_states = plot(1:T, μ_true, label="Sann latent variabel μ_t", lw=3, color=:black, ls=:dash)
plot!(plot_states, 1:T, gibbs_mean_mu,
      ribbon=(gibbs_mean_mu .- gibbs_lower_ci, gibbs_upper_ci .- gibbs_mean_mu),
      label="Gibbs Smoothed Mean (95% CI)", lw=2, color=:red, fillalpha=0.2)
plot!(plot_states, 1:T, bq_mean,
      ribbon=(bq_mean .- bq_lower_ci, bq_upper_ci .- bq_mean),
      label="BQ/BO Smoothed Mean (95% CI)", lw=2, color=:blue, fillalpha=0.2)
scatter!(plot_states, 1:T, y_obs, label="Observerad data y_t", alpha=0.5, ms=2, color=:grey)
title!("Jämförelse av Smoothed States μ_t")
xlabel!("Tid (t)")

# --- Visa plots ---
display(plot_params)
display(plot_states)

println("\nJämförelse klar.")
println("BQ/BO MAP-skattning: $(round.(theta_map_bq, digits=3))")
println("Gibbs posterior mean: $(round.(gibbs_mean_params, digits=3))")
println("Sanna parametrar: $(round.(true_log_params, digits=3))")


################
# ==============================================================================
# DEL 6: Jämför Marginalfördelningarna för Hyperparametrar
# ==============================================================================

println("\nGenererar jämförelseplott för marginalfördelningar...")

# --- 1. Extrahera Gibbs-samples för log-varianser ---
# (Dessa bör redan finnas från föregående plott)
log_s2_eps_gibbs = log.(gibbs_results.σ2_ε)
log_s2_eta_gibbs = log.(gibbs_results.σ2_η)


# --- 2. Beräkna marginalfördelningar från BQ/BO-metodens GP-yta ---

# posterior_surface_bq är en (100x100)-matris med posterior-densitet på ett rutnät
# grid1 motsvarar log(σ²_ε) och grid2 motsvarar log(σ²_η)

# För att få marginalen för log(σ²_ε), integrerar (summerar) vi ut log(σ²_η)
# Detta motsvarar att summera längs varje rad i matrisen
marginal_log_s2_eps_bq = vec(sum(posterior_surface_bq, dims=2))

# För att få marginalen för log(σ²_η), integrerar (summerar) vi ut log(σ²_ε)
# Detta motsvarar att summera längs varje kolumn i matrisen
marginal_log_s2_eta_bq = vec(sum(posterior_surface_bq, dims=1))

# Normalisera marginalerna så att de är sanna densiteter (integrerar till 1)
dx_eps = grid1[2] - grid1[1]
dx_eta = grid2[2] - grid2[1]
marginal_log_s2_eps_bq ./= (sum(marginal_log_s2_eps_bq) * dx_eps)
marginal_log_s2_eta_bq ./= (sum(marginal_log_s2_eta_bq) * dx_eta)


# --- 3. Skapa jämförande plots ---

# Plott för log(σ²_ε)
p_eps = density(log_s2_eps_gibbs,
    label="Gibbs Posterior",
    linewidth=2.5,
    color=:red,
    title="Marginalfördelning för log(σ²_ε)")
plot!(p_eps, grid1, marginal_log_s2_eps_bq,
    label="BQ/BO Marginal",
    linewidth=2.5,
    color=:blue,
    ls=:dash)
vline!(p_eps, [true_log_params[1]],
    label="Sann Värde",
    color=:black,
    ls=:dot,
    linewidth=2)
xlabel!(p_eps, "log(σ²_ε)")
ylabel!(p_eps, "Densitet")

# Plott för log(σ²_η)
p_eta = density(log_s2_eta_gibbs,
    label="Gibbs Posterior",
    linewidth=2.5,
    color=:red,
    title="Marginalfördelning för log(σ²_η)")
plot!(p_eta, grid2, marginal_log_s2_eta_bq,
    label="BQ/BO Marginal",
    linewidth=2.5,
    color=:blue,
    ls=:dash)
vline!(p_eta, [true_log_params[2]],
    label="Sann Värde",
    color=:black,
    ls=:dot,
    linewidth=2)
xlabel!(p_eta, "log(σ²_η)")

# Kombinera till en figur
plot_marginals = plot(p_eps, p_eta, layout=(1, 2), size=(1200, 500), legend=:topleft)

# --- Visa plotten ---
display(plot_marginals)