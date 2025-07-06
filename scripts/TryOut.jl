

using GaussianProcesses
using Random
using Optim
using Distributions
using Plots

# Set random seed for reproducibility
Random.seed!(123)

# Define black-box objective function (you can change this!)
f(x) = sin(3x) + 0.5x^2 - x + 0.2randn()
fNf(x) = sin(3x) + 0.5x^2 - x 

# Rescaling functions used for GP to ensure inputs are in a suitable range. (Easier to set lengthscales and mor robust optimization).
rescale(x, lo, hi) = 2 * (x .- lo) ./ (hi - lo) .- 1
inv_rescale(x, lo, hi) = 0.5 * (x .+ 1) * (hi - lo) .+ lo

# Bounds
lo, hi = -5.0, 5.0

# Initial design points (column vector form)
X = reshape([-2.5, 0.0, 2.0], :, 1)
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
KB = [[-3, -8], [3, 8]];
NB = [-6., 3.];

lo=-5.; hi=5.;

# Number of optimization iterations
nIter = 1

modelSettings = (mean=mean1, kernel = kernel1, logNoise = logNoise1, 
                 kernelBounds = KB, noiseBounds=NB, xdim=d, xBounds=[lo, hi]

)

optimizationSettings = (nIter=3, ξ=0.2,  n_restarts=20, bounds=(-1.0, 1.0), acq=expected_improvement)


warmStart = (X, y)
warmStart = (XO, yO)

gpO, XO, yO = BO(f, modelSettings, optimizationSettings, warmStart)

##################

# Create a dense grid in the original input space
x_plot = range(lo, hi, length=200);
x_plot_array = collect(x_plot);  # needed for broadcasting

# Rescale to [-1, 1]
x_plot_scaled = rescale(x_plot_array, lo, hi);
x_plot_scaled_matrix = reshape(x_plot_scaled, 1, :);

# Predict from GP
μ, σ² = predict_y(gpO, x_plot_scaled_matrix);

# Plot in original coordinates
plot(x_plot, μ; ribbon=sqrt.(σ²), label="GP prediction", lw=2, xlab);
plot!(x_plot, fNf.(x_plot), label="True function f(x)", lw=2);
xlabel!("x")
ylabel!("f(x)")
title!("GP vs True Function")
scatter!(XO, yO, label="Samples", color=:black)

###################

