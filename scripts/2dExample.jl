#########################
# 2d case
# Set random seed for reproducibility
Random.seed!(123)

# Bivariate Gaussian mixture pdf (objective function)
function gaussian_mixture_pdf(x::Vector{Float64})
    μ1 = [-2.0, -2.0]
    Σ1 = [2.0 1.5; 1.5 2.0]
    μ2 = [2.0, 2.0]
    Σ2 = [2.0 -1.7; -1.7 2.0]
    p1 = MvNormal(μ1, Σ1)
    p2 = MvNormal(μ2, Σ2)
    w1, w2 = 0.5, 0.5
    return 10*(w1 * pdf(p1, x) + w2 * pdf(p2, x))
end

f(x) = gaussian_mixture_pdf(x) + 0.05randn()




# Posterior minimum only on observed points
function posteriorMinObs(gp, X)
    μ, _ = predict_f(gp, X')
    return minimum(μ)
end



# Initialize parameters
d = 2
lower=[-5,-7]
upper=[6,7]
bounds = (lower, upper)

# Initial design: 5 random points in 2D within bounds
# Sample 5 points in d dimensions
X = hcat([rand(Uniform(l, u), 5) for (l, u) in zip(bounds[1], bounds[2])]...) 
y = [f(vec(X[i, :])) for i in 1:size(X, 1)]


d=2
mean1 = MeanConst(0.0)
kernel1 = Mat52Ard(log(1.)*ones(d), 1.0)  # lengthscale zeros means will be optimized
logNoise1 = log(1e-1)              # low noise since pdf is deterministic
KB = [[-3, -3., -5], [log(1.5), log(1.5), 2]];
NB = [-6., 1.];


modelSettings = (mean=mean1, kernel = kernel1, logNoise = logNoise1, 
                 kernelBounds = KB, noiseBounds=NB, xdim=d, xBounds=bounds

)


# The results are quite sensitive to the tuning parameter in the 2d case. 
# Especially ucb is sensitive.
optimizationSettings = (nIter=3, tuningPar=0.1,  n_restarts=30, acq=expected_improvement)

#optimizationSettings = (nIter=20, tuningPar=3.5,  n_restarts=20,acq=upper_confidence_bound)

warmStart = (X, y)
warmStart = (XO, yO)
gpO, XO, yO, objMin, obsMin = BO(f, modelSettings, optimizationSettings, warmStart)


# Create grid for plotting
xx = range(bounds[1][1], bounds[2][1], length=100)
yy = range(bounds[1][2], bounds[2][2], length=100)

# Compute true function on grid
Z_true = [gaussian_mixture_pdf([x,y]) for y in yy, x in xx]  # (y,x) indexing for heatmap

# Predict GP mean on grid
grid_points = [[x, y] for y in yy, x in xx]
X_grid = reduce(hcat, grid_points)  # 2 x 10000
μ, _ = predict_f(gpO, rescale(X_grid', bounds[1], bounds[2])')
Z_gp = reshape(μ, 100, 100)'

# Plot true function heatmap
p1 = heatmap(xx, yy, -Z_true, title="True function (Gaussian Mixture PDF)", xlabel="x", ylabel="y")

# Plot GP mean heatmap with samples overlaid
p2 = heatmap(xx, yy, Z_gp', title="GP mean posterior", xlabel="x", ylabel="y")
scatter!(p2, XO[:,1], XO[:,2], color=:white, markersize=5, label="Samples")
scatter!(p2, [XO[end,1]], [XO[end,2]], color=:red, markersize=8, label="Last samples")

plot(p1, p2, layout=(1,2), size=(1000,400))






# Noisy heatmap, how data looks:
ZNoise = [-f([x,y]) for y in yy, x in xx]  # (y,x) indexing for heatmap
p1 = heatmap(xx, yy, ZNoise, title="True function (Gaussian Mixture PDF)", xlabel="x", ylabel="y")


########################
# Define the input domain from -1 to 1
x_domain = -1.0:0.005:1.0
x_matrix = reshape(x_domain, :, 1) # Reshape for GaussianProcesses.jl

# Number of random functions to draw from each GP prior
n_samples = 3

# --- 2. Create the Two Gaussian Process Priors ---

# GP 1: Small Lengthscale (expects a "wiggly" function)
l_small = 0.01
kernel_small = SE(log(l_small), 0.0) # SE(log(lengthscale), log(signal_variance))
# FIX: Use the GP() constructor for creating a data-free prior.
# It only needs a mean function and a kernel.
gp_small = GP([0.;;], [0.], MeanZero(), kernel_small)

# GP 2: Large Lengthscale (expects a "smooth" function)
l_large = 2.0
kernel_large = SE(log(l_large), 0.0)
# FIX: Use the GP() constructor here as well.
gp_large = GP([0.;;], [0.], MeanZero(), kernel_large)


# --- 3. Draw Samples from Each GP Prior ---

# The rand() function works directly with the GP object.
# Draw random functions from the "wiggly" GP
samples_small_l = rand(gp_small, x_matrix', n_samples)

# Draw random functions from the "smooth" GP
samples_large_l = rand(gp_large, x_matrix', n_samples)


# --- 4. Plot the Results for Comparison ---

# Create the plot canvas
plot(legend=:topright, title="GP Prior Samples with Different Lengthscales",
     xlabel="Input Domain (x)", ylabel="Function Value (f(x))", size=(1200,600))

# Plot the samples from the small lengthscale GP (wiggly)
plot!(x_domain, samples_small_l,
      linewidth=1,
      label="ℓ = $l_small (Wiggly)",
      color=:lightblue)

# Plot the samples from the large lengthscale GP (smooth)
plot!(x_domain, samples_large_l,
      linewidth=2,
      linestyle=:solid,
      label="ℓ = $l_large (Smooth)",
      color=:red)

# Display the final plot
current()


####################
x_domain = -1.0:0.01:1.0
x_matrix = reshape(x_domain, :, 1) # Omforma för GaussianProcesses.jl

# Antal slumpmässiga funktioner att dra från varje GP-prior
n_samples = 3

# Håll längdskalan konstant för att isolera effekten av signalvariansen
l_fixed = 0.5

# --- 2. Skapa två Gaussiska process-priors med olika signalvarians ---

# GP 1: Liten signalvarians (låg amplitud)
signal_variance_small = 0.5
# Kerneln tar logaritmen av parametrarna: SE(log(lengthscale), log(signal_variance))
kernel_small_sf = SE(log(l_fixed), log(signal_variance_small))
gp_small_sf = GP([0.;;], [0.],MeanZero(), kernel_small_sf)

# GP 2: Stor signalvarians (hög amplitud)
signal_variance_large = 2.0
kernel_large_sf = SE(log(l_fixed), log(signal_variance_large))
gp_large_sf = GP([0.;;], [0.],MeanZero(), kernel_large_sf)


# --- 3. Dra sampel från varje GP-prior ---

# Dra slumpmässiga funktioner från GP:n med låg amplitud
samples_small_sf = rand(gp_small_sf, x_matrix', n_samples)

# Dra slumpmässiga funktioner från GP:n med hög amplitud
samples_large_sf = rand(gp_large_sf, x_matrix', n_samples)


# --- 4. Plotta resultaten för jämförelse ---

# Skapa plotten
plot(legend=:topright, title="GP Prior-sampel med olika signalvarians (σ_f)",
     xlabel="Indomän (x)", ylabel="Funktionsvärde (f(x))")

# Plotta sampel från GP:n med liten signalvarians (låg amplitud)
plot!(x_domain, samples_small_sf,
      linewidth=2,
      linestyle=:dash,
      label="σ_f = $signal_variance_small (Låg amplitud)",
      color=:green)

# Plotta sampel från GP:n med stor signalvarians (hög amplitud)
plot!(x_domain, samples_large_sf,
      linewidth=2,
      linestyle=:solid,
      label="σ_f = $signal_variance_large (Hög amplitud)",
      color=:purple)

# Visa den slutgiltiga plotten
current()
