

using GaussianProcesses
using Random, Optim, Distributions, Plots
using BOOP

# Set random seed for reproducibility
Random.seed!(123)

# Define black-box objective function (you can change this!)
f(x) = -(1.5*sin(3x) + 0.5x^2 - x + 0.2randn())
fNf(x) = -(1.5*sin(3x) + 0.5x^2 - x )


# Bounds
lo, hi = -4.0, 5.0

# Initial design points (column vector form)
X = reshape([-2.5, 0.0, 4.0], :, 1)
y = f.(X)
Xscaled= rescale(X, lo, hi)

# Define GP components
mean1 = MeanConst(0.0)
kernel1 = Mat52Ard([0.0], 1.0)
logNoise1 = log(0.1)


d=1
mean1 = MeanConst(0.0)
kernel1 = Mat52Ard(3*ones(d), 1.0)  # lengthscale zeros means will be optimized
logNoise1 = log(1e-1)              # low noise since pdf is deterministic
KB = [[-3, -5], [3, 3]];
NB = [-6., 3.];

lo=-4.; hi=5.;

# Number of optimization iterations
nIter = 1

modelSettings = (mean=mean1, kernel = kernel1, logNoise = logNoise1, 
                 kernelBounds = KB, noiseBounds=NB, xdim=d, xBounds=[lo, hi]

)

optimizationSettings = (nIter=1, tuningPar=0.02,  n_restarts=20, acq=expected_improvement)
optimizationSettings = (nIter=1, tuningPar=1.,  n_restarts=20, acq=upper_confidence_bound)



warmStart = (X, y[:])
warmStart = (XO, yO)

gpO, XO, yO, objMin, obsMin, postMaxObsY  = BO(f, modelSettings, optimizationSettings, warmStart)

    

##################
# Probably make plots to one function and put it in a "plotUtils" script.
# Create a dense grid in the original input space
x_plot = range(lo, hi, length=200);
x_plot_array = collect(x_plot);  # needed for broadcasting

# Rescale to [-1, 1]
x_plot_scaled = rescale(x_plot_array, lo, hi);
x_plot_scaled_matrix = reshape(x_plot_scaled, 1, :);

# Predict from GP
μScaled, σ²Scaled = predict_f(gpO, x_plot_scaled_matrix);
μ_y = mean(yO)
σ_y = max(std(yO), 1e-6)


# Transform predictions back to the original scale
μ = vec(μScaled) .* σ_y .+ μ_y
σ² = vec(σ²Scaled) .* (σ_y^2)


# Plot in original coordinates
plot(x_plot, μ; ribbon=sqrt.(σ²), label="GP prediction", lw=2, xlabel="x");
plot!(x_plot, fNf.(x_plot), label="True function f(x)", lw=2);
xlabel!("x")
ylabel!("f(x)")
title!("GP vs True Function")
scatter!(XO, yO, label="Samples", color=:black)

vline!([obsMin], label="Observed Maximizer", color=:red, linestyle=:dash)
vline!([objMin], label="Objective Maximizer", color=:blue, linestyle=:dash)
hline!([postMaxObsY], label=" Posterior Maximum", color=:green, linestyle=:dash)
###################

xn=-0.9
fM=maximum(yO)
expected_improvement(gpO, xn, fM; ξ = 0.10)
upper_confidence_bound(gpO, xn; κ = 2.0)
@time knowledgeGradientMonteCarlo(gpO, xn; n_samples=500)
knowledgeGradientDiscrete(gpO, xn, Matrix(reshape(evalGrid,1,:)))
@time knowledgeGradientHybrid(gpO, xn, n_z=15)



evalGrid = -1.1:0.05:1
EIContainer = []
UCBContainer = []
KGContainer = []
KGDiscreteContainer = []
KGHybridContainer = []
for i in evalGrid
    push!(EIContainer, expected_improvement(gpO, i, fM; ξ = 0.10))
end

for i in evalGrid
    push!(UCBContainer, upper_confidence_bound(gpO, i; κ = 2.0))
end

using ProgressMeter
@showprogress for i in evalGrid
    push!(KGContainer, knowledgeGradientMonteCarlo(gpO, i; n_samples=500))
end

evalGrid2 = -1.1:0.1:1
@showprogress for i in evalGrid2
    push!(KGDiscreteContainer, knowledgeGradientDiscrete(gpO, i, Matrix(reshape(evalGrid2,1,:))))
end

@showprogress for i in evalGrid
    push!(KGHybridContainer, knowledgeGradientHybrid(gpO, i, n_z=50))    
end




pEI = plot(evalGrid, EIContainer, label="Expected Improvement", xlabel="x", ylabel="EI value", title="Acquisition Functions")
vline!([evalGrid[argmax(EIContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(EIContainer)])")
pUBC=plot(evalGrid, UCBContainer, label="Upper Confidence Bound", xlabel="x", ylabel="UCB value")
vline!([evalGrid[argmax(UCBContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(UCBContainer)])")
pKG=plot(evalGrid, KGContainer, label="Knowledge Gradient", xlabel="x", ylabel="KG value")
vline!([evalGrid[argmax(KGContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(KGContainer)])")
pKGD=plot(evalGrid2, KGDiscreteContainer, label="Discrete Knowledge Gradient", xlabel="x", ylabel="KG value")
vline!([evalGrid2[argmax(KGDiscreteContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid2[argmax(KGDiscreteContainer)])")
pKGH=plot(evalGrid, KGHybridContainer, label="Hybrid Knowledge Gradient", xlabel="x", ylabel="KG value")
vline!([evalGrid[argmax(KGHybridContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(KGHybridContainer)])")



plot(pEI, pUBC,  pKGD, plot(gpO, xlim=(-1,1)), layout=(4,1), size=(800,900))

gpO.logNoise.value=log(.1)
plot(gpO)
gpO.logNoise.value=log(100)
plot(gpO)

# Sätt brusvärdet och plotta som vanligt
gpO.logNoise.value = log(0.1)
p1 = plot(gpO, title="Lågt Brus (logNoise = log(0.1))")


# --- Andra plotten (högt brus) ---
# Skapa ett HELT NYTT GP-objekt med det höga brusvärdet
# Vi återanvänder data och kernel från det gamla objektet
gp_high_noise = GP(gpO.x, gpO.y, gpO.mean, gpO.kernel, log(.05))
p2 = plot(gp_high_noise, title="Högt Brus (logNoise = log(100))")






evalGrid = -1.:0.05:1
EIContainer = []
UCBContainer = []
KGContainer = []
KGDiscreteContainer = []
KGHybridContainer = []
for i in evalGrid
    push!(EIContainer, expected_improvement(gp_high_noise, i, fM; ξ = 0.10))
end

for i in evalGrid
    push!(UCBContainer, upper_confidence_bound(gp_high_noise, i; κ = 2.0))
end

using ProgressMeter
@showprogress for i in evalGrid
    push!(KGContainer, knowledgeGradientMonteCarlo(gp_high_noise, i; n_samples=50))
end

evalGrid2 = -1.:0.01:1
for i in evalGrid2
    push!(KGDiscreteContainer, knowledgeGradientDiscrete2(gp_high_noise, i, Matrix(reshape(evalGrid2,1,:))))
end

@showprogress for i in evalGrid
    push!(KGHybridContainer, knowledgeGradientHybrid(gp_high_noise, i, n_z=10))
end




pEI = plot(evalGrid, EIContainer, label="Expected Improvement", xlabel="x", ylabel="EI value", title="Acquisition Functions")
vline!([evalGrid[argmax(EIContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(EIContainer)])")
pUBC=plot(evalGrid, UCBContainer, label="Upper Confidence Bound", xlabel="x", ylabel="UCB value")
vline!([evalGrid[argmax(UCBContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(UCBContainer)])")
pKG=plot(evalGrid, KGContainer, label="Knowledge Gradient", xlabel="x", ylabel="KG value")
vline!([evalGrid[argmax(KGContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(KGContainer)])")
pKGD=plot(evalGrid2, KGDiscreteContainer, label="Discrete Knowledge Gradient", xlabel="x", ylabel="KG value")
vline!([evalGrid2[argmax(KGDiscreteContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid2[argmax(KGDiscreteContainer)])")
pKGH=plot(evalGrid, KGHybridContainer, label="Hybrid Knowledge Gradient", xlabel="x", ylabel="KG value")
vline!([evalGrid[argmax(KGHybridContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(KGHybridContainer)])")



plot(pEI, pUBC, pKG, pKGD, layout=(4,1), size=(800,900))

pKG=plot(evalGrid, KGContainer, label="Knowledge Gradient", xlabel="x", ylabel="KG value")
vline!([evalGrid[argmax(KGContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(KGContainer)])")
plot!(evalGrid2, reverse(KGDiscreteContainer./2), label="Discrete Knowledge Gradient", xlabel="x", ylabel="KG value")
vline!([evalGrid2[argmax(KGDiscreteContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid2[argmax(KGDiscreteContainer)])")




gp_high_noise = GP(gpO.x, gpO.y, gpO.mean, gpO.kernel, log(.005))
p2 = plot(gp_high_noise, title="Högt Brus (logNoise = log(100))")


μ_current, _ = predict_f(gp_high_noise, evalGrid)


# Get the predictive distribution of a noisy observation y at xnew.
xn=0.6
_, σ²_y_new = predict_y(gp_high_noise, [xn;;])
σ_y_new = sqrt(max(σ²_y_new[1], 1e-8))
# Calculate the covariance vector between the domain points and xnew.


K_domain_xnew = cov(gp_high_noise.kernel, Matrix(reshape(evalGrid,1,:)), [xn;;])
#plot(gp_high_noise)
plot!(evalGrid, K_domain_xnew)



# This vector describes the change in the posterior mean.
#update_vector = K_domain_xnew ./ σ²_y_new[1]
# Construct μ and σ for the core algorithm
μ = vec(μ_current)
#σ = vec(update_vector) .* σ_y_new
σ = vec(K_domain_xnew / σ_y_new)
plot!(evalGrid, σ, lw=3)





###############
gp_high_noise = GP(gpO.x, gpO.y, gpO.mean, gpO.kernel, log(.05))
p2 = plot(gp_high_noise, title="Högt Brus (logNoise = log(100))")
μ_current, _ = predict_f(gp_high_noise, evalGrid)

# Get the predictive distribution of a noisy observation y at xnew.
xn=0.6
_, σ²_y_new = predict_y(gp_high_noise, [xn;;])
σ_y_new = sqrt(max(σ²_y_new[1], 1e-8))
# Calculate the covariance vector between the domain points and xnew.

K_domain_xnew = cov(gp_high_noise.kernel, Matrix(reshape(evalGrid,1,:)), [xn;;])
plot!(evalGrid, K_domain_xnew)

μ = vec(μ_current)
σ = vec(K_domain_xnew ./ σ_y_new)
plot!(evalGrid, σ, lw=3)

println("σ stats - min: $(minimum(σ)), max: $(maximum(σ)), mean: $(mean(σ))")
println("σ_y_new: ", σ_y_new)


p3 = plot(gp_high_noise, title="Högt Brus (logNoise = log(100))")
μ_current, _ = predict_f(gp_high_noise, evalGrid)

# Get the predictive distribution of a noisy observation y at xnew.
xn=-0.2
_, σ²_y_new = predict_y(gp_high_noise, [xn;;])
σ_y_new = sqrt(max(σ²_y_new[1], 1e-8))
# Calculate the covariance vector between the domain points and xnew.

K_domain_xnew = cov(gp_high_noise.kernel, Matrix(reshape(evalGrid,1,:)), [xn;;])
plot!(evalGrid, K_domain_xnew)

μ = vec(μ_current)
σ = vec(K_domain_xnew / σ_y_new)
plot!(evalGrid, σ, lw=3)

expected_max_future = ExpectedMaxGaussian(μ, σ)
maximum(μ)

plot(p2,p3, layout=(2,1))


knowledgeGradientDiscrete(gp_high_noise, 0.6, Matrix(reshape(evalGrid2,1,:)))
knowledgeGradientDiscrete(gp_high_noise, -0.2, Matrix(reshape(evalGrid2,1,:)))



function posterior_cov(gp::GPE, X1::AbstractMatrix, X2::AbstractMatrix)
    # Formeln är: k_n(X1, X2) = k(X1, X2) - k(X1, X) * K_inv * k(X, X2)
    # där K = k(X,X) + σ_n²*I
    
    # Beräkna de priora kovarianstermerna
    prior_cov_12 = cov(gp.kernel, X1, X2)
    prior_cov_1X = cov(gp.kernel, X1, gp.x)
    prior_cov_X2 = cov(gp.kernel, gp.x, X2)
    
    # Lös systemet K * M = k(X, X2) för att effektivt få K_inv * k(X, X2)
    # GaussianProcesses.jl lagrar den Cholesky-faktoriserade matrisen i gp.cK
    # vilket gör detta mycket effektivt.
    # L*L' * M = prior_cov_X2  =>  L' * M = L \ prior_cov_X2  =>  M = L' \ (L \ prior_cov_X2)
    K_inv_k_X2 = gp.cK \ prior_cov_X2
  
    
    # Beräkna korrektionstermen
    correction = prior_cov_1X * K_inv_k_X2
    
    return prior_cov_12 - correction
end


# --- KGD-funktionen med den korrekta beräkningen ---
"""
Beräknar den diskreta Knowledge Gradient.
Denna version använder den korrekta posteriora kovariansen.
"""
function knowledgeGradientDiscrete2(gp::GPE, xnew, domain_points::Matrix{Float64})
    xvec = xnew isa Number ? [xnew] : xnew
    xnew_mat = reshape(xvec, :, 1)

    # Hämta prediktiv distribution för en brusig observation
    _, σ²_y_new = predict_y(gp, xnew_mat)
    

    std_dev_y_new = sqrt(σ²_y_new[1])

    # --- DEN KORREKTA BERÄKNINGEN AV ~σ ---
    # Använd den nya funktionen för att få den posteriora kovariansen
    post_cov_vec = posterior_cov(gp, domain_points, xnew_mat)
    sigma_tilde = post_cov_vec ./ std_dev_y_new
    
    # Hämta nuvarande posteriora medelvärde
    μ_current, _ = predict_f(gp, domain_points)
    
    μ = vec(μ_current)
    σ = vec(sigma_tilde)
    
    expected_max_future = ExpectedMaxGaussian(μ, σ)
    max_μ_current = maximum(μ)
    kg_value = expected_max_future - max_μ_current
    
    return kg_value
end

predict_y(gpO, xnew_mat)