

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
optimizationSettings = (nIter=5, tuningPar=2.,  n_restarts=20, acq=upper_confidence_bound)



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

xn=-1.
fM=maximum(yO)
expected_improvement(gpO, xn, fM; ξ = 0.10)
upper_confidence_bound(gpO, xn; κ = 2.0)
@time knowledgeGradientMonteCarlo(gpO, xn; n_samples=4200)
knowledgeGradientDiscrete(gpO, xn, Matrix(reshape(evalGrid,1,:)))

evalGrid = -1.:0.01:1
EIContainer = []
UCBContainer = []
KGContainer = []
KGDiscreteContainer = []
for i in evalGrid
    push!(EIContainer, expected_improvement(gpO, i, fM; ξ = 0.10))
end

for i in evalGrid
    push!(UCBContainer, upper_confidence_bound(gpO, i; κ = 2.0))
end

using ProgressMeter
@showprogress for i in evalGrid
    push!(KGContainer, knowledgeGradientMonteCarlo(gpO, i; n_samples=300))
end

evalGrid2 = -1.:0.001:1
for i in evalGrid2
    push!(KGDiscreteContainer, knowledgeGradientDiscrete(gpO, i, Matrix(reshape(evalGrid2,1,:))))
end




pEI = plot(evalGrid, EIContainer, label="Expected Improvement", xlabel="x", ylabel="EI value", title="Acquisition Functions")
vline!([evalGrid[argmax(EIContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(EIContainer)])")
pUBC=plot(evalGrid, UCBContainer, label="Upper Confidence Bound", xlabel="x", ylabel="UCB value")
vline!([evalGrid[argmax(UCBContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(UCBContainer)])")
pKG=plot(evalGrid, KGContainer, label="Knowledge Gradient", xlabel="x", ylabel="KG value")
vline!([evalGrid[argmax(KGContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(KGContainer)])")
pKGD=plot(evalGrid2, KGDiscreteContainer, label="Discrete Knowledge Gradient", xlabel="x", ylabel="KG value")
vline!([evalGrid2[argmax(KGDiscreteContainer)]], linestyle=:dash, color=:black, label="maximum at $(evalGrid[argmax(KGDiscreteContainer)])")

plot(pEI, pUBC, pKG, pKGD, layout=(4,1), size=(800,900))