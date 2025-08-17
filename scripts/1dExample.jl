

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

# Optimization settings
optimizationSettings = (nIter=1, tuningPar=0.02,  n_restarts=20, acq=expected_improvement, nSim=nothing, nq=nothing, dmp=nothing)
optimizationSettings = (nIter=1, tuningPar=2.0,  n_restarts=20, acq=upper_confidence_bound, nSim=nothing, nq=nothing, dmp=nothing)
optimizationSettings = (nIter=1, tuningPar=nothing,  n_restarts=10, acq=knowledgeGradientHybrid, nq=60, dmp=nothing)
optimizationSettings = (nIter=1, tuningPar=nothing,  n_restarts=10, acq=knowledgeGradientDiscrete, nSim=nothing, nq=nothing, dmp=Matrix(reshape(-1:0.01:1., 1, :)))


warmStart = (X, y[:])
warmStart = (XO, yO)

@time gpO, XO, yO, objMin, obsMin, postMaxObsY  = BO(f, modelSettings, optimizationSettings, warmStart);


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
