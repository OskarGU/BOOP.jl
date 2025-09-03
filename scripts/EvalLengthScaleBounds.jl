# In this script we try to figure out suitable bounds when optimizing the kernel hyperparameters and variance.
# for BO. Since I rescale the data to be in -1,1 it is not useful to allow for too large/small lengthscales.
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