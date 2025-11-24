using GaussianProcesses
using Random, Optim, Distributions, Plots, Statistics, LinearAlgebra
using BOOP 


Random.seed!(41)

# Problembeskrivning:
# x[1] (Kontinuerlig): Ska optimeras i [-1, 1]
# x[2] (Diskret):      Ska optimeras i [1, 13] (heltal)
# Globalt Max ligger vid: x ≈ 0.5, z = 7.

function true_function(x_vec)
    x_cont = x_vec[1]
    z_int  = round(x_vec[2]) # Tvinga till heltal (om optimeraren fuskar)
    
    # Diskret del: En "Gauss-klocka" som är högst vid z=7
    disc_part = exp(-(z_int - 7)^2 / 8.0)
    
    # Kontinuerlig del: En cosinus-våg som är högst vid x=0.5
    cont_part = sqrt(abs(x_cont))*cos(3 * (x_cont*1.2 - 0.5))
    
    # Lite brus för att göra det realistiskt
    noise = 0.05 * randn()
    
    # Skala upp med 10
    return 10.0 * disc_part * cont_part + noise
end

# En ren version utan brus för plottning
f_true_plot(x, z) = 10.0 * exp(-(z - 7)^2 / 8.0) * sqrt(abs(x))*cos(3 * (x*1.2 - 0.5)) 

# Gränser
x_lo, x_hi = -1.0, 1.0
z_lo, z_hi = 1, 13
d = 2

# Generera startdata (Warm Start)
N_init = 5
X_warm = zeros(N_init, d)
X_warm[:, 1] = rand(Uniform(x_lo, x_hi), N_init)    # Kontinuerlig
X_warm[:, 2] = rand(z_lo:z_hi, N_init)              # Diskret (Heltal!)
y_warm = [true_function(row) for row in eachrow(X_warm)]

warmStart = (X_warm, y_warm)

# ==============================================================================
# 2. KONFIGURATION AV BO
# ==============================================================================

# 1. Skapa din Kernel (Här antar jag att GarridoMerchanKernel finns i BOOP)
# Vi använder Matern52Ard som baskernel.
base_k = Mat52Ard([0.1;0.6], 0.8) 
gm_kernel = BOOP.GarridoMerchanKernel(base_k, [2], [z_lo:z_hi])

# 2. Modellinställningar
# Notera: deepcopy(gm_kernel) är viktigt så vi inte delar minne felaktigt!
modelSettings = (
    mean = MeanConst(0.0),
    kernel = deepcopy(gm_kernel),   
    logNoise = -2.0,                
    kernelBounds = [[-3.0, 0.5, -2.0], [3.0, 3.0, 5.0]], 
    noiseBounds = [-5.0, 1.0],
    xdim = d,
    # Gränserna för rescale:
    xBounds = ([x_lo, x_hi]) 
)

# 3. Optimeringsinställningar
opt_settings = OptimizationSettings(
    nIter = 3,          
    n_restarts = 13,
    acq_config = EIConfig(ξ=0.1) # Exploitative EI
    #acq_config = UCBConfig(κ=2.5)
    #acq_config = KGHConfig(n_z=50)
)

# ==============================================================================
# 3. KÖR BAYESIAN OPTIMIZATION
# ==============================================================================
println("--- Startar BO med Garrido-Merchán Kernel ---")

# VIKTIGT: DiscreteKern=1 anger att sista dimensionen INTE ska skalas om till [-1, 1].
# Den behålls som heltal (1..13) så att kerneln kan jobba med den.
warmStart = (X_final, y_final)
@time gp, X_final, y_final, max_x, max_val, max_obs_x, max_obs_val = BO(
    true_function, modelSettings, opt_settings, warmStart; DiscreteKern=1
)

println("\nGlobal Model Max hittat vid: $max_x")
println("Värde vid max: $(round(max_val, digits=3))")

# ==============================================================================
# 4. VISUALISERING (Heatmaps & Slices)
# ==============================================================================
println("Skapar plottar...")

# Skapa grid för plottning
xs = range(x_lo, x_hi, length=100)
zs = z_lo:z_hi

# Hämta statistik för att kunna "skala tillbaka" y-värden manuellt för plotten
μ_y, σ_y = mean(y_final), std(y_final)


# --- A. HEATMAPS (Sanning vs Modell) ---
Z_true = [f_true_plot(x, z) for z in zs, x in xs] # Matris för sanningen

Z_model = zeros(length(zs), length(xs))

# Loopa igenom gridet för att skapa modellens heatmap
for (i, z) in enumerate(zs)
    for (j, x) in enumerate(xs)
        # 1. Konstruera punkten. 
        # Eftersom vi körde DiscreteKern=1, så är GP:n tränad på:
        # Dim 1: Skalad [-1, 1] (vilket x redan är här)
        # Dim 2: Oskalad [1, 13] (vilket z är)
        pt = [x, Float64(z)] 
        
        # 2. Prediktera (ger skalat mean)
        μ_scaled, _ = predict_f(gp, reshape(pt, d, 1))
        
        # 3. Skala tillbaka y
        Z_model[i, j] = μ_scaled[1] * σ_y + μ_y
    end
end

p1 = heatmap(xs, zs, Z_true, title="Sanning", xlabel="x (Kont)", ylabel="z (Diskret)", c=:viridis);
p2 = heatmap(xs, zs, Z_model, title="GP Modell", xlabel="x (Kont)", ylabel="z (Diskret)", c=:viridis);
# Lägg till våra samplade punkter i modell-plotten
scatter!(p2, X_final[:,1], X_final[:,2], label="Sampel", mc=:red, ms=4, legend=false);
plot(p1,p2, size=(1400,400));

# --- B. SNITT (Slices) VID OLIKA Z ---
# Vi tittar på hur funktionen ser ut för tre specifika diskreta val
z_slices = [4, 7, 11] # 7 är optimum
slice_plots = []

for z_val in z_slices
    # Sanningen för detta z
    y_slice_true = [f_true_plot(x, z_val) for x in xs]
    
    # Modellens gissning (med osäkerhetsband)
    # Skapa en hel rad med punkter för detta z
    pts = hcat(collect(xs), fill(float(z_val), length(xs)))'
    
    μ_pred_sc, σ2_pred_sc = predict_f(gp, pts)
    σ_pred_sc = sqrt.(max.(σ2_pred_sc, 0.0))
    
    # Skala tillbaka
    y_slice_model = μ_pred_sc .* σ_y .+ μ_y
    σ_slice_model = σ_pred_sc .* σ_y
    
    # Kolla vilka punkter vi faktiskt har samplat på detta z-plan
    is_at_z = (round.(X_final[:, 2]) .== z_val)
    samples_x = X_final[is_at_z, 1]
    samples_y = y_final[is_at_z]

    p = plot(xs, y_slice_true, label="Sanning (z=$z_val)", lw=2, lc=:black, title="Snitt vid z = $z_val", ylims=(-10, 12))
    plot!(p, xs, y_slice_model, label="GP Mean", lw=2, lc=:blue, ribbon=(1.96*σ_slice_model, 1.96*σ_slice_model), fillalpha=0.2, fc=:blue)
    scatter!(p, samples_x, samples_y, label="Observerat", mc=:red, ms=6)
    push!(slice_plots, p)
end;

# --- C. SLUTGILTIG PLOT ---
l = @layout [a b; c d e]
final_plot = plot(p1, p2, slice_plots..., layout=l, size=(1200, 1000));
display(final_plot);


# Vi varierar Z kontinuerligt för att se stegen tydligt
z_plot_grid = range(0, 13, length=500)

# Vi väljer 4 olika värden för den KONTINUERLIGA variabeln (x)
x_fixed_values = [0.1, 0.9, 0.5, 0.9]

# Layout 2x2
plot_layout = @layout [a b; c d]
p_steps = plot(layout=plot_layout, size=(1000, 800), legend=:topleft);

for (idx, x_val) in enumerate(x_fixed_values)
    
    # Skapa input-matris (2 x N_punkter)
    # Rad 1: Fixerat x-värde
    # Rad 2: Varierande z-värde (kontinuerligt grid)
    X_test_slice = zeros(2, length(z_plot_grid))
    X_test_slice[1, :] .= BOOP.rescale(x_val, [-1.], [1.]; integ=0)
    X_test_slice[2, :] .= z_plot_grid

    # Predicera
    μScaled, σ²Scaled = predict_f(gp, X_test_slice)
    μ = μScaled*σ_y .+ μ_y
    σ = sqrt.(σ²Scaled)*σ_y
    # Konfidensintervall
    lower = μ .- 1.96 .* σ
    upper = μ .+ 1.96 .* σ

    # Plotta GP (Trappstegen)
    plot!(p_steps, z_plot_grid, μ, 
          ribbon=(μ .- lower, upper .- μ),
          fillalpha=0.2,
          lw=2,
          label="GP (Trappsteg)",
          title="Fixerat x = $(x_val)",
          xlabel="Diskret variabel (z)",
          ylabel="Output",
          subplot=idx,
          color=idx
    )

    # Plotta "Sanningen" (för att se hur bra stegen matchar)
    # Vi beräknar den sanna funktionen för varje z i gridden
    y_true = [true_function([x_val, z]) for z in z_plot_grid]
    plot!(p_steps, z_plot_grid, y_true, 
          label="Sann funktion", 
          linestyle=:dash, 
          color=:black, 
          alpha=0.6,
          subplot=idx
    )

    # Lägg till träningsdata som råkar ligga nära detta x-värde
    # Vi visar punkter där x är inom +/- 0.25 från vårt fixerade värde
    mask = abs.(X_final'[1, :] .- x_val) .< 0.05
    
    if any(mask)
        scatter!(p_steps, X_final'[2, mask], y_final[mask],
                 label="Data (nära x=$(x_val))",
                 markercolor=:red,
                 markersize=5,
                 subplot=idx
        )
    end
end;

#display(p_steps);

p3 = plot(plot(p_steps[1]),
plot(p_steps[2]),
plot(p_steps[3]),
layout=(1,3));

pHeat = plot(p1, p2);
pSlice = plot(slice_plots..., layout=(1,3));
plot(pHeat,pSlice,p3,layout=(3,1), size=(1500,1200))