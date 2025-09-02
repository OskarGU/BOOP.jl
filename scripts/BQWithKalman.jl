using LinearAlgebra, Distributions, BOOP, SMCsamplers

"""
    kalman_loglikelihood(U, Y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀)

Computes the total log-likelihood using the original, unmodified 
kalmanfilter_update function.
"""
function kalman_loglikelihood(U, Y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀)
    # --- Initialisering ---
    T = size(Y, 1)
    q = size(U, 2)
    r = size(Y, 2)
    staticA = (ndims(A) == 3) ? false : true
    staticC = (ndims(C) == 3) ? false : true
    staticΣₑ = (ndims(Σₑ) == 3) ? false : true
    staticΣₙ = (ndims(Σₙ) == 3) ? false : true

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)
    total_logL = 0.0

    # --- Huvudloop ---
    for t = 1:T
        # Hämta data och parametrar för tidpunkt t
        At = staticA ? A : @view A[:, :, t]
        Ct = staticC ? C : @view C[:, :, t]
        Σₑt = staticΣₑ ? Σₑ : @view Σₑ[:, :, t]
        Σₙt = staticΣₙ ? Σₙ : @view Σₙ[:, :, t]
        u = (q == 1) ? U[t] : U[t, :]
        y = (r == 1) ? Y[t] : Y[t, :]

        # 1. Anropa din originalfunktion för att filtrera och få prediktionerna
        μ_new, Σ_new, μ̄, Σ̄ = kalmanfilter_update(μ, Σ, u, y, At, B, Ct, Σₑt, Σₙt)

        # 2. Använd returvärdena μ̄ och Σ̄ för att beräkna log-likelihooden
        y_pred_mean = Ct * μ̄
        y_pred_cov = Ct * Σ̄ * Ct' + Σₑt
        logL_t = logpdf(MvNormal(vec(y_pred_mean), Symmetric(y_pred_cov)), [y])

        total_logL += logL_t

        # 3. Rulla framåt till nästa steg
        μ = μ_new
        Σ = Σ_new
    end

    return total_logL
end
#################################3


mutable struct LGSSParams
    a::Float64
    σᵥ::Float64
    σₑ::Float64
end

prio(θ) = Normal(0, 10*θ.σᵥ / √((1 - θ.a^2)));
transition(θ, state, t) = Normal(θ.a * state, θ.σᵥ);
observation(θ, state, t) = Normal(state, θ.σₑ);

a = 0.9         # Persistence
σᵥ = 0.3        # State std deviation
σₑ = 0.5        # Observation std deviation
θ = LGSSParams(a, σᵥ, σₑ); # Set up parameter struct for PGAS


T = 200     # Length of time series
x = zeros(T)
y = zeros(T)
x0 = rand(prio(θ))
for t in 1:T
    if t == 1
        x[t] = rand(transition(θ, x0, t))
    else
        x[t] = rand(transition(θ, x[t-1], t))
    end
    y[t] = rand(observation(θ, x[t], t))
end
plot(x; label="state, x", xlabel="t", lw = 2, legend = :topleft, color = 3)
plot!(y; seriestype=:scatter, label="observed, y", xlabel="t", markersize = 2,
    color = 1, markerstrokecolor = :auto 
)




# Set up the LGSS for FFBS and sample
Σₑ = [θ.σₑ^2]
Σₙ = [θ.σᵥ^2]
μ₀ = [0;;]
Σ₀ = 10^2*[θ.σᵥ^2/(1-θ.a^2);;]
A = θ.a
C = 1
B = 0
U = zeros(T,1);
Nₛ=100
using SMCsamplers
FFBSdraws = FFBS(U, y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀, Nₛ);FFBSmean = mean(FFBSdraws, dims = 3)
FFBSquantiles = quantile_multidim(FFBSdraws, [0.025, 0.975], dims = 3);



function logLikKal(Σₑ)
kalman_loglikelihood(U, y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀)
end

ranΣ = 0.1:0.01:1.5
len= length(ranΣ)
ll = []
for i in 1:len
    push!(ll, logLikKal([ranΣ[i]]))
end

plot(ranΣ, ll, label="Log-Likelihood", xlabel="Σₑ", ylabel="Log-Likelihood", lw=2)
vline!([Σₑ])

logLikKal([0.1])





##################################
function sample_Σₑ_posterior(y, x_trajectory, α₀, β₀)
    T = length(y)
    
    # a) Beräkna summan av kvadratiska fel (Sum of Squared Errors)
    errors = y .- x_trajectory
    SSE = sum(errors.^2)
    
    # b) Uppdatera hyperparametrarna för Inverse Gamma-posteriorn
    αₙ = α₀ + T / 2
    βₙ = β₀ + SSE / 2
    
    # c) Dra och returnera ett nytt värde för Σₑ från dess posteriorfördelning
    return rand(InverseGamma(αₙ, βₙ))
end


# --- Den uppdaterade Gibbs-samplern ---

function gibbs_sampler_lgss(y, θ_fixed, initial_Σₑ, prior_params, n_draws)
    # Packa upp fasta parametrar och prior-parametrar
    a = θ_fixed.a
    σᵥ = θ_fixed.σᵥ
    α₀, β₀ = prior_params
    
    T = length(y)
    
    # Initiera lagringsarrayer
    x_draws = zeros(T, n_draws)
    Σₑ_draws = zeros(n_draws)
    
    # Sätt startvärden
    Σₑ_current = initial_Σₑ
    
    # Sätt upp de fasta delarna av FFBS-anropet
    U = zeros(T, 1)
    C = 1.0
    B = 0.0
    Σₙ = [σᵥ^2]
    μ₀ = [0.0;;]
    Σ₀ = [10^2 * σᵥ^2 / (1 - a^2);;]
    
    println("Startar Gibbs-sampling...")
    for k in 1:n_draws
        if k % 500 == 0
            println("Iteration $k / $n_draws")
        end
        
        # --- Steg 1: Sampla x | y, Σₑ ---
        A = a
        Σₑ_mat = [Σₑ_current]
        x_new_trajectory = FFBS(U, y, A, B, C, Σₑ_mat, Σₙ, μ₀, Σ₀, 1)[2:end, 1, 1]
        
        # --- Steg 2: Sampla Σₑ | y, x (nu med vår nya funktion) ---
        # All komplex logik är nu inkapslad i en egen funktion.
        Σₑ_current = sample_Σₑ_posterior(y, x_new_trajectory, α₀, β₀)
        
        # --- Lagra resultaten ---
        x_draws[:, k] .= x_new_trajectory
        Σₑ_draws[k] = Σₑ_current
    end
    
    println("Gibbs-sampling avslutad!")
    return x_draws, Σₑ_draws
end


# --- Sätt upp och kör (denna del är exakt samma som förut) ---

# Hyperparametrar för prior
α₀ = 2.01
β₀ = 0.5
prior_params = (α₀, β₀)

# Modell och data


# Sampler-inställningar
n_draws = 15000
burn_in = 1000
θ_fixed = LGSSParams(a, σᵥ, 0.0)
initial_Σₑ = 1.0

# Kör samplaren!
x_draws, Σₑ_draws = gibbs_sampler_lgss(y, θ_fixed, initial_Σₑ, prior_params, n_draws)

# Analysera...
# ... (all din plot-kod fungerar precis som förut) ...
Σₑ_posterior = Σₑ_draws[burn_in+1:end]
x_posterior_mean = mean(x_draws[:, burn_in+1:end], dims=2)

p1 = plot(Σₑ_draws, label="Trace of Σₑ", xlabel="Iteration", ylabel="Σₑ"); vline!([burn_in], label="Burn-in", color=:red, linestyle=:dash)
p2 = histogram(Σₑ_posterior, normalize=:pdf, label="Posterior of Σₑ", bins=50); vline!([θ_true.σₑ^2], label="True value = $(round(θ_true.σₑ^2, digits=2))", color=:red, lw=2)
p3 = plot(x, label="True State", xlabel="t", lw=2, color=3); plot!(y, seriestype=:scatter, label="Observations", markersize=2, color=1); plot!(x_posterior_mean, label="Smoothed State (Posterior Mean)", color=:orange, lw=2)
plot(p1, p2, p3, layout=(3,1), size=(800, 700))

