"""
    systematic_resampler(weights)

Utför systematisk resampling.
Returnerar en vektor med föräldra-index (ancestor indices).
"""
function systematic_resampler(weights::AbstractVector)
    N = length(weights)
    indices = zeros(Int, N)
    
    # Generera en startpunkt från en uniform fördelning
    u_start = rand() / N
    
    # Skapa en kumulativ summa av vikterna
    cdf = cumsum(weights)
    
    j = 1
    for i in 1:N
        # Gå framåt i cdf tills vi hittar rätt intervall
        while u_start > cdf[j]
            j += 1
        end
        indices[i] = j
        u_start += 1/N
    end
    return indices
end

################
"""
    auxiliary_particle_filter(y, p, N, θ, prior, transition, observation)

Kör ett Auxiliary Particle Filter (APF) med `N` partiklar.
Returnerar en uppskattning av log-sannolikheten `log p(y|θ)` samt de filtrerade partiklarna och vikterna.

Funktionen använder `systematic_resampler` som standard.

Argument:
- `y`: Vektor med observationer (T element).
- `p`: Dimensionaliteten av tillståndsrymden (state space).
- `N`: Antal partiklar.
- `θ`: Struct med modellparametrar.
- `prior`: Funktion som returnerar en fördelning för p(x₁).
- `transition`: Funktion som returnerar p(xₜ | xₜ₋₁, θ).
- `observation`: Funktion som returnerar p(yₜ | xₜ, θ).
"""
function auxiliary_particle_filter(y, p, N, θ, prior_dist, transition, observation)
    T = length(y)
    
    # Initiera lagringsmatriser
    particles = zeros(N, p, T)      # Partiklar: X
    weights = zeros(N, T)           # Vikter: w
    
    log_likelihood = 0.0

    # --- Tidpunkt t = 1 ---
    # Sampla initiala partiklar från priorn
    initial_particles = rand(prior_dist(θ), N)
    if p == 1
        particles[:, 1, 1] .= initial_particles
    else
        particles[:, :, 1] .= initial_particles'
    end

    # Beräkna initiala vikter
    log_w = zeros(N)
    for n in 1:N
        log_w[n] = logpdf(observation(θ, particles[n, :, 1]), y[1])
    end
    
    max_log_w = maximum(log_w)
    w_norm = exp.(log_w .- max_log_w)
    sum_w = sum(w_norm)
    
    weights[:, 1] .= w_norm ./ sum_w
    log_likelihood += max_log_w + log(sum_w) - log(N)

    # --- Tidpunkter t = 2...T ---
    for t in 2:T
        # --- APF Steg 1: Beräkna "first-stage" vikter och resampling ---
        # Dessa vikter baseras på hur väl varje partikel från t-1 förväntas
        # förklara observationen vid tid t.
        
        first_stage_log_w = zeros(N)
        # För varje partikel, dra en prediktion och evaluera sannolikheten
        for n in 1:N
            # Här kan man använda medelvärdet av transition-fördelningen som en "lookahead"
            # För en enkel Local Level-modell är transition(x_prev) bara N(x_prev, σ²), så medelvärdet är x_prev.
            x_prev = particles[n, :, t-1]
            predicted_state_mean = mean(transition(θ, x_prev)) # Enkel lookahead
            first_stage_log_w[n] = logpdf(observation(θ, predicted_state_mean), y[t])
        end

        max_log_w_fs = maximum(first_stage_log_w)
        w_fs_unnorm = weights[:, t-1] .* exp.(first_stage_log_w .- max_log_w_fs)
        w_fs_norm = w_fs_unnorm ./ sum(w_fs_unnorm)
        
        # Resampla baserat på dessa "first-stage" vikter
        ancestor_indices = systematic_resampler(w_fs_norm)
        
        # --- APF Steg 2: Propagera de valda partiklarna ---
        for n in 1:N
            parent_particle = particles[ancestor_indices[n], :, t-1]
            particles[n, :, t] .= rand(transition(θ, parent_particle))
        end

        # --- APF Steg 3: Beräkna de slutgiltiga vikterna ---
        final_log_w = zeros(N)
        for n in 1:N
            # Vi måste korrigera för lookahead-approximationen
            x_t = particles[n, :, t]
            x_t_minus_1_resampled = particles[ancestor_indices[n], :, t-1]
            predicted_state_mean = mean(transition(θ, x_t_minus_1_resampled))
            
            log_obs_t = logpdf(observation(θ, x_t), y[t])
            log_obs_t_mean = logpdf(observation(θ, predicted_state_mean), y[t])
            
            final_log_w[n] = log_obs_t - log_obs_t_mean
        end

        max_log_w = maximum(final_log_w)
        w_norm = exp.(final_log_w .- max_log_w)
        sum_w = sum(w_norm)

        weights[:, t] .= w_norm ./ sum_w
        log_likelihood += max_log_w + log(sum_w) - log(N)
    end
    
    return log_likelihood, particles, weights
end


##################
# Run ex:
# ANTAGANDE: arma_reparam(state; ztrans="monahan") finns definierad
# ANTAGANDE: p = 2 (för en AR(2)-modell)

# Prior för det initiala tillståndet ϕ₀
# θ används inte här, men behålls för ett konsekvent API
prior_dist(θ) = MvNormal(zeros(p), 1.0 * I) # En vag prior

# Tillståndsövergången för ϕ_t
function transition(θ, x_prev)
    # Packa upp Σ-komponenterna från θ och skapa matrisen
    # θ = [H, Σ₁₁, Σ₂₂, Σ₁₂]
    Σ = [θ[2]  θ[4];
         θ[4]  θ[3]]
    return MvNormal(x_prev, Σ)
end

# Observationssannolikheten p(y_t | ϕ_t)
# Detta är en "wrapper" runt din befintliga funktion
function observation(θ, x_curr, y_t, lags_yt)
    # Packa upp H från θ
    H = θ[1]
    
    # Reparametrisera till stationära AR-koefficienter
    ar_coeffs, _ = arma_reparam(x_curr; ztrans="monahan")
    
    # Beräkna predikterat medelvärde
    predicted_mean = lags_yt' * ar_coeffs
    
    return Normal(predicted_mean, sqrt(H))
end


function estimate_log_likelihood_tvar(θ, y_data, p; N=500)
    if any(θ[[1,2,3]] .<= 0) || (θ[2]*θ[3] - θ[4]^2 <= 0) # H > 0, Σ > 0
        return -Inf
    end
    
    T = length(y_data)
    
    # Skapa laggade versioner av y för observation-funktionen
    X_lags = [y_data[t-1:-1:t-p] for t in p+1:T]
    y_to_predict = y_data[p+1:end]
    
    # Skapa anonyma funktioner med rätt signatur för APF
    # Notera att vi måste hantera de laggade y-värdena
    obs_func = (theta, state, t) -> observation(theta, state, y_to_predict[t], X_lags[t])

    # Kör APF på den del av datan där vi har tillräckligt med laggar
    log_likelihood, _, _ = auxiliary_particle_filter(y_to_predict, p, N, θ, prior_dist, transition, obs_func)
    
    return log_likelihood
end

# --- Inställningar och körning av BQ ---

# Definiera prior-gränser för de 4 parametrarna [H, Σ₁₁, Σ₂₂, Σ₁₂]
parameter_bounds = (
    [0.01, 0.001, 0.001, -0.1],  # Undre gränser
    [2.0,  0.5,   0.5,    0.1]   # Övre gränser
)

# Integranden som BQ anropar. Den hanterar omskalning och priors.
function integrand_for_bq_tvar(θ_unscaled, y_data, p, bounds)
    lo = bounds[1]
    hi = bounds[2]
    θ = vec(inv_rescale(reshape(θ_unscaled, 1, :), lo, hi))
    
    prior_volume = prod(hi .- lo)
    prior_prob = 1.0 / prior_volume # Uniform prior

    log_likelihood = estimate_log_likelihood_tvar(θ, y_data, p)
    
    if isinf(log_likelihood) return 0.0 end
    
    return exp(log_likelihood) * prior_prob
end

# Skapa funktionen `f` som BQ ska integrera
f_for_bq_tvar = θ_scaled -> integrand_for_bq_tvar(θ_scaled, y_data, p, parameter_bounds)

# Modellinställningar för BQ (nu en 4D-funktion)
modelSettings_bq_tvar = (
    xdim = 4,
    xBounds = ([-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]),
    mean = MeanZero(),
    kernel = SE(0.0, 0.0),
    logNoise = -5.0, # Justera baserat på partikelfiltrets varians
    kernelBounds = [fill(-2.0, 4), fill(2.0, 4)],
    noiseBounds = [-8., 0.] 
)

# Samplingsinställningar för BQ (använder din adaptiva metod)
optimizationSettings_tvar = (
    nIter = 50, # En 4D-rymd kräver fler punkter!
    n_restarts = 15,
    acq = width_in_original_scale,
    dmp = nothing,
    c_base = 1.96,
    c_sensitivity = 0.5
)

# Startpunkter
X_start_scaled = rand(10, 4) .* 2.0 .- 1.0 # Fler startpunkter är bra
y_start = [f_for_bq_tvar(X_start_scaled[i, :]) for i in 1:size(X_start_scaled, 1)]
warmStart_bq_tvar = (X_start_scaled, vec(y_start))

println("Startar BQ för parametrar i tidsvarierande AR(2)-modell...")

# Kör BQ-loopen!
# final_gp_tvar, _, _, integral_val_tvar, _ = 
#     BQ_WSABI(f_for_bq_tvar, modelSettings_bq_tvar, optimizationSettings_tvar, warmStart_bq_tvar)

println("BQ-körning avslutad!")
# println("Uppskattad modellevidens p(y): ", integral_val_tvar)




##################
##################
#####################
# --- Nödvändiga paket ---
using GaussianProcesses
using Optim
using Distributions
using Plots
using LinearAlgebra # För I (identitetsmatris)

# ---------------------------------------------------------------------------
# --- Steg 1: Din fullständiga BQ- och PF-kod ---
# ---------------------------------------------------------------------------

# Hjälpfunktioner för omskalning
function rescale(X, lo, hi)
    return 2 .* (X .- lo') ./ (hi' .- lo') .- 1
end
inv_rescale(X_scaled, lo, hi) = ((X_scaled .+ 1) ./ 2) .* (hi' .- lo') .+ lo'

# Acquisition-funktioner
function width_in_original_scale end
function posterior_variance(gp, xnew) # Lägger till denna för flexibilitet
    xvec = xnew isa Number ? [xnew] : xnew
    xvec = reshape(xvec, :, 1)
    _, σ² = predict_f(gp, xvec)
    return σ²[1]
end

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

# Din `propose_next` anpassad för BQ (kallar den propose_nextBQ för tydlighet)
function propose_nextBQ(gp; n_restarts=20, acq, dmp, stats=nothing)
    d = gp.dim
    best_acq_val = -Inf
    best_x = zeros(d)

    function objective_to_minimize(x)
        val = 0.0
        if acq == posterior_variance
             val = posterior_variance(gp, x)
        elseif acq == width_in_original_scale
            if stats === nothing || length(stats) < 3
                error("width_in_original_scale requires 'stats' (mean, std, c) to be provided.")
            end
            val = original_scale_uncertainty(gp, x; y_mean=stats[1], y_std=stats[2], c=stats[3])
        else
            error("Unknown acquisition function: $acq")
        end
        return -val
    end

    for _ in 1:n_restarts
        x0 = rand(Uniform(-1., 1.), d)
        res = optimize(objective_to_minimize, -1.0 * ones(d), 1.0 * ones(d), x0, Fminbox(LBFGS()); autodiff = :forward)
        current_acq_val = -res.minimum
        if current_acq_val > best_acq_val
            best_acq_val = current_acq_val
            best_x = res.minimizer
        end
    end
    return best_x
end

# Din `estimate_integral_wsabi` (bytt namn till `estimate_model_evidence` för tydlighet)
function estimate_model_evidence(gp, bounds; n_samples=100_000, y_mean=0.0, y_std=1.0)
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

# Huvudloopen för BQ
function BQ_for_SSM(f, modelSettings, optimizationSettings, warmStart)
    X_scaled, y = warmStart # Notera: vi jobbar direkt med skalade X-värden
    
    for i in 1:optimizationSettings.nIter
        y_warped = log.(y .+ 1e-9)
        μ_yw = mean(y_warped)
        σ_yw = max(std(y_warped), 1e-6)
        y_warped_scaled = (y_warped .- μ_yw) ./ σ_yw

        gp = GP(X_scaled', y_warped_scaled, modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
        optimize!(gp; kernbounds=modelSettings.kernelBounds, noisebounds=modelSettings.noiseBounds, options=Optim.Options(iterations=100))

        c_dynamisk = optimizationSettings.c_base # Start med basvärdet för c
        # (Logik för adaptivt c kan läggas till här om så önskas)

        x_next_scaled = propose_nextBQ(gp;
                                     n_restarts=optimizationSettings.n_restarts,
                                     acq=optimizationSettings.acq,
                                     dmp=optimizationSettings.dmp,
                                     stats=(μ_yw, σ_yw, c_dynamisk))

        y_next = f(x_next_scaled)

        X_scaled = vcat(X_scaled, x_next_scaled')
        y = vcat(y, y_next)
        
        println("Iter $i: Nästa parameterförslag (skalat): $(round.(x_next_scaled, digits=3)), Likelihood*Prior: $(round(y_next, digits=5))")
    end
    
    y_warped_final = log.(y .+ 1e-9)
    μ_yw_final = mean(y_warped_final)
    σ_yw_final = max(std(y_warped_final), 1e-6)
    y_ws_final = (y_warped_final .- μ_yw_final) ./ σ_yw_final

    gp_final = GP(X_scaled', y_ws_final, modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
    optimize!(gp_final; kernbounds=modelSettings.kernelBounds, noisebounds=modelSettings.noiseBounds, options=Optim.Options(iterations=500))

    # Byt namn på BQ-integranden, eftersom vi inte använder WSABI-warping på den
    # BQ beräknar integralen av f(θ) = p(y|θ)p(θ) direkt
    integral_estimate = estimate_model_evidence(gp_final, modelSettings.xBounds, n_samples=200_000, y_mean=mean(y), y_std=max(std(y), 1e-6))

    return gp_final, X_scaled, y, integral_estimate, (μ_yw_final, σ_yw_final)
end

# Partikelfilter-kod
function systematic_resampler(weights::AbstractVector)
    N = length(weights)
    indices = zeros(Int, N)
    u_start = rand() / N
    cdf = cumsum(weights)
    j = 1
    for i in 1:N
        while u_start > cdf[j]
            j += 1
        end
        indices[i] = j
        u_start += 1/N
    end
    return indices
end

function auxiliary_particle_filter(y, p, N, θ, prior_dist, transition, observation)
    T = length(y)
    particles = zeros(N, p, T)
    weights = zeros(N, T)
    log_likelihood = 0.0

    # ... (resten av APF-koden från tidigare svar) ...
    # (Denna del är korrekt och behöver inga ändringar)
    initial_particles = rand(prior_dist(θ), N)
    if p == 1; particles[:, 1, 1] .= initial_particles; else; particles[:, :, 1] .= initial_particles'; end
    log_w = [logpdf(observation(θ, particles[n, :, 1], 1), y[1]) for n in 1:N]
    max_log_w = maximum(log_w)
    w_norm = exp.(log_w .- max_log_w)
    sum_w = sum(w_norm)
    weights[:, 1] .= w_norm ./ sum_w
    log_likelihood += max_log_w + log(sum_w) - log(N)
    for t in 2:T
        first_stage_log_w = zeros(N)
        for n in 1:N
            predicted_state_mean = mean(transition(θ, particles[n, :, t-1]))
            first_stage_log_w[n] = logpdf(observation(θ, predicted_state_mean, t), y[t])
        end
        max_log_w_fs = maximum(first_stage_log_w)
        w_fs_unnorm = weights[:, t-1] .* exp.(first_stage_log_w .- max_log_w_fs)
        w_fs_norm = w_fs_unnorm ./ sum(w_fs_unnorm)
        ancestor_indices = systematic_resampler(w_fs_norm)
        for n in 1:N
            particles[n, :, t] .= rand(transition(θ, particles[ancestor_indices[n], :, t-1]))
        end
        final_log_w = zeros(N)
        for n in 1:N
            x_t = particles[n, :, t]
            x_t_minus_1_resampled = particles[ancestor_indices[n], :, t-1]
            predicted_state_mean = mean(transition(θ, x_t_minus_1_resampled))
            log_obs_t = logpdf(observation(θ, x_t, t), y[t])
            log_obs_t_mean = logpdf(observation(θ, predicted_state_mean, t), y[t])
            final_log_w[n] = log_obs_t - log_obs_t_mean
        end
        max_log_w = maximum(final_log_w)
        w_norm = exp.(final_log_w .- max_log_w)
        sum_w = sum(w_norm)
        weights[:, t] .= w_norm ./ sum_w
        log_likelihood += max_log_w + log(sum_w) - log(N)
    end
    return log_likelihood, particles, weights
end

# ---------------------------------------------------------------------------
# --- Steg 2: Sätt upp och kör ditt specifika TV-AR(2) problem ---
# ---------------------------------------------------------------------------

println("Definierar TV-AR(2) modell och genererar data...")

# ANTAGANDE: Du har en funktion `arma_reparam` tillgänglig.
# Vi skapar en dummy-funktion här för att göra koden körbar.
arma_reparam(state; ztrans) = (state, 1.0)
p = 2 # AR(2)

# Modellfunktioner
prior_dist(θ) = MvNormal(zeros(p), 1.0 * I)
function transition(θ, x_prev)
    Σ = [θ[2]  θ[4]; θ[4]  θ[3]]
    return MvNormal(x_prev, Σ)
end
function observation(θ, x_curr, t, y_data, p)
    lags_yt = y_data[t-1:-1:t-p]
    H = θ[1]
    ar_coeffs, _ = arma_reparam(x_curr; ztrans="monahan")
    predicted_mean = lags_yt' * ar_coeffs
    return Normal(predicted_mean, sqrt(H))
end

# Generera syntetisk data
true_params = [0.5, 0.01, 0.01, 0.001] # [H, Σ₁₁, Σ₂₂, Σ₁₂]
T = 100
y_data = zeros(T)
phi_true = zeros(T+1, p)
phi_true[1,:] = rand(prior_dist(true_params))
for t in 1:T
    if t > p
        lags = y_data[t-1:-1:t-p]
        ar_coeffs, _ = arma_reparam(phi_true[t,:]; ztrans="monahan")
        mean_y = lags' * ar_coeffs
        y_data[t] = rand(Normal(mean_y, sqrt(true_params[1])))
    else
        y_data[t] = randn() # Fyll på med lite slumpdata i början
    end
    phi_true[t+1,:] = rand(transition(true_params, phi_true[t,:]))
end

# "Black Box"-funktioner
function estimate_log_likelihood_tvar(θ, y_data, p; N=500)
    if any(θ[[1,2,3]] .<= 0) || (θ[2]*θ[3] - θ[4]^2 <= 0)
        return -Inf
    end
    y_to_filter = y_data[p+1:end]
    obs_func = (theta, state, t_filt) -> observation(theta, state, t_filt+p, y_data, p)
    log_likelihood, _, _ = auxiliary_particle_filter(y_to_filter, p, N, θ, prior_dist, transition, obs_func)
    return log_likelihood
end

parameter_bounds = ([0.01, 0.001, 0.001, -0.05], [2.0, 0.1, 0.1, 0.05])

# ANVÄND DENNA ISTÄLLET för integrand_for_bq_tvar och f_for_bq_tvar

function log_integrand_for_bq_tvar(θ_unscaled, y_data, p, bounds)
    lo = bounds[1]
    hi = bounds[2]
    θ = vec(inv_rescale(reshape(θ_unscaled, 1, :), lo, hi))
    
    prior_volume = prod(hi .- lo)
    log_prior_prob = -log(prior_volume) # Log of uniform prior probability

    log_likelihood = estimate_log_likelihood_tvar(θ, y_data, p; N=500)
    
    # Returnera logaritmen av integranden
    # Om log_likelihood är -Inf, returnera ett mycket litet tal
    if isinf(log_likelihood)
        return -1e6 # Ett stort negativt tal istället för -Inf
    end
    
    return log_likelihood + log_prior_prob
end

# Den nya funktionen som BQ ska bygga en modell av
f_log_for_bq = θ_scaled -> log_integrand_for_bq_tvar(θ_scaled, y_data, p, parameter_bounds)

# --- KÖRNING ---
# Denna definition måste ske EFTER att y_data finns
f_for_bq_tvar = θ_scaled -> integrand_for_bq_tvar(θ_scaled, y_data, p, parameter_bounds)

modelSettings_bq_tvar = (xdim = 4, xBounds = ([.001,.001,.001,.001], [1.0,1.0,1.0,1.0]), mean = MeanZero(), 
                         kernel = SE(0.0, 0.0), logNoise = -8.0, 
                         kernelBounds = [fill(-3.0, 4), fill(3.0, 4)], noiseBounds = [-12., 0.])

modelSettings_bq_tvar = (
    xdim = 4,
    xBounds = ([-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]),
    mean = MeanZero(),
    
    # --- ÄNDRING 1: Byt till SEArd-kärna ---
    # Skapa en kernel med en separat längdskala för varje av de 4 dimensionerna.
    kernel = SEArd(zeros(4), 0.0),

    logNoise = -5.0, 
    
    # --- ÄNDRING 2: Uppdatera gränserna för att matcha 5 hyperparametrar ---
    # 4 för längdskalorna + 1 för signalvariansen
    kernelBounds = [fill(-3.0, 5), fill(3.0, 5)], 
    
    noiseBounds = [-8., 0.]
)

optimizationSettings_tvar = (nIter = 50, n_restarts = 15, acq = width_in_original_scale,
                           dmp = nothing, c_base = 1.96, c_sensitivity = 0.5)

println("Skapar startpunkter...")
X_start_scaled = rand(10, 4) .* 2.0 .- 1.0
y_start = [f_log_for_bq(X_start_scaled[i, :]) for i in 1:size(X_start_scaled, 1)]
warmStart_bq_tvar = (X_start_scaled, vec(y_start))

println("Startar Bayesiansk Kvadratur för parameterinferens...")
# Byt namn på BQ_WSABI till BQ_for_SSM för att undvika förvirring
final_gp_tvar, X_hist_tvar, y_hist_tvar, integral_val_tvar, _ = 
    BQ_for_SSM(f_for_bq_tvar, modelSettings_bq_tvar, optimizationSettings_tvar, warmStart_bq_tvar)

println("\nBQ-körning avslutad!")
println("Uppskattad modellevidens p(y): ", integral_val_tvar)
println("Sanna parametrar var: [H, Σ₁₁, Σ₂₂, Σ₁₂] = ", true_params)

# Hitta MAP-estimatet från surrogatmodellen
# (Kräver en `posteriorMax`-funktion som optimerar GP:ns medelvärde)
# _, map_params_scaled = posteriorMax(final_gp_tvar) 
# map_params_original = vec(inv_rescale(reshape(map_params_scaled, 1, :), parameter_bounds...))
# println("MAP-estimat för parametrarna: ", round.(map_params_original, digits=4))