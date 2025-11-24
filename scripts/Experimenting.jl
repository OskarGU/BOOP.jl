
# Rewrite rescale to take discrete vars.
function rescale(X, lo, hi; integ=0)
    # X is n×d matrix, lo and hi are length-d vectors
    if integ == 0
        return 2 .* (X .- lo') ./ (hi' .- lo') .- 1
    else
        return hcat(2 .* (X[:,1:end-integ] .- lo') ./ (hi' .- lo') .- 1, [round.(X[:,end]);])
    end
end

Xr = 10*rand(8,3)
Xd = [1:8;]

Xreg = rescale(hcat(Xr,Xd), [0.1, 1, 0.3, 0.8], [8, 12, 14, 18])
Xdisc = rescale(hcat(Xr,Xd), [0.1, 1, 0.3], [8, 12, 14], integ=1)

lob = [0.1, 1, 0.3, 0.8]
hib = [8, 12, 14, 18]
lo= [0.1, 1, 0.3]
hi = [8, 12, 14]


function inv_rescale(XScaled, lo, hi; integ = 0) 
    if integ == 0
        ((XScaled .+ 1) ./ 2) .* (hi' .- lo') .+ lo'
    else
        hcat(((XScaled[:,1:end-integ] .+ 1) ./ 2) .* (hi' .- lo') .+ lo', [round.(XScaled[:,end]);])
    end
end

inv_rescale(Xreg, lob, hib, integ=0)
inv_rescale(Xdisc, lo, hi; integ = 1)  


# This functions makes a continuous search at each value of the discrete variable. 
# This can be very costly if several discrete varaibles,

using Optim
# Notera: lade till X, Y, M, P i where-satsen och tog bort T
function propose_nextt(gp::GPE{X,Y,M,K,P}, f_max; n_restarts::Int, acq_config::BOOP.AcquisitionConfig) where {X, Y, M, P, K<:BOOP.GarridoMerchanKernel}
    print("G-M search")
    # 1. Hämta info direkt från kernel-objektet
    disc_dims = gp.kernel.integer_dims
    disc_ranges = gp.kernel.integer_ranges

    d = gp.dim
    cont_dims = setdiff(1:d, disc_dims)

    # 2. Skapa alla kombinationer av de diskreta värdena
    discrete_combinations = vec(collect(Iterators.product(disc_ranges...)))

    full_objective = BOOP._get_objective(gp, f_max, acq_config)
    best_acq_val = -Inf
    best_x_full = zeros(d)

    # 3. Exhaustive Splitting Loop
    for d_vals in discrete_combinations

        # Wrapper för att bara optimera de kontinuerliga delarna
        function sub_objective(x_cont)
            x_full = zeros(d)
            if !isempty(cont_dims)
                x_full[cont_dims] = x_cont
            end
            x_full[disc_dims] .= d_vals # Fixera de diskreta
            return full_objective(x_full)
        end

        # Optimera kontinuerliga variabler (om de finns)
        if !isempty(cont_dims)
            n_cont = length(cont_dims)
            starts = [-ones(n_cont) .+ (ones(n_cont) .- -ones(n_cont)) .* ((i .+ 0.5) ./ n_restarts) for i in 0:(n_restarts - 1)]

            for i in 1:n_restarts
                x0 = starts[i]
                # Optimera inom boxen [-1, 1] för de kontinuerliga
                res = optimize(sub_objective, -1.0 * ones(n_cont), 1.0 * ones(n_cont), x0, Optim.Fminbox(LBFGS()))

                curr_val = -Optim.minimum(res)
                if curr_val > best_acq_val
                    best_acq_val = curr_val
                    best_x_full[cont_dims] = Optim.minimizer(res)
                    best_x_full[disc_dims] .= d_vals
                end
            end
        else
            # Endast diskreta variabler (inga kontinuerliga att optimera)
            val = -sub_objective(Float64[])
            if val > best_acq_val
                best_acq_val = val
                best_x_full[disc_dims] .= d_vals
            end
        end
    end

    return best_x_full
end


function propose_nextt(gp, f_max; n_restarts::Int, acq_config::BOOP.AcquisitionConfig)
    print("regular")
    d = gp.dim
    best_acq_val = -Inf
    best_x = zeros(d)

    # Dispatch to the correct helper to get the objective function
    objective_to_minimize = BOOP._get_objective(gp, f_max, acq_config)
    starts =  [-ones(d) .+ (ones(d) .- -ones(d)) .* ((i .+ 0.5) ./ n_restarts) for i in 0:(n_restarts - 1)]
    for i in 1:n_restarts
        x0 =  starts[i]#rand(Uniform(-1., 1.), d)

        # Use the type to select the optimizer
        res = if acq_config isa BOOP.KnowledgeGradientConfig # Use the abstract type
            optimize(objective_to_minimize, -1.0, 1.0, x0, Fminbox(NelderMead()))
        else
            optimize(objective_to_minimize, -1.0 * ones(d), 1.0 * ones(d), x0, Fminbox(LBFGS()); autodiff = :forward)
        end

        current_acq_val = -Optim.minimum(res)
        if current_acq_val > best_acq_val
            best_acq_val = current_acq_val
            best_x = Optim.minimizer(res)
        end
    end
    return best_x
end



#### For dispatcing to match G-M:
function posteriorMax(gp::GPE{X,Y,M,K,P}; n_starts=20) where {X,Y,M,P,K<:BOOP.GarridoMerchanKernel}
    # Återanvänd logiken från propose_next men optimera Posterior Mean direkt!
    print("Special")
    disc_dims = gp.kernel.integer_dims
    disc_ranges = gp.kernel.integer_ranges
    d = gp.dim
    cont_dims = setdiff(1:d, disc_dims)
    
    # Hämta posterior mean funktionen
    # Notera: predict_f returnerar (mean, variance), vi vill ha mean[1]
    acq_mean(x) = predict_f(gp, reshape(x, d, 1))[1][1]

    best_val = -Inf
    best_x_full = zeros(d)
    
    discrete_combinations = vec(collect(Iterators.product(disc_ranges...)))

    for d_vals in discrete_combinations
        
        # Sub-funktion för kontinuerlig optimering
        function sub_mean(x_cont)
             T = eltype(x_cont)
             x_full = zeros(T, d)
             if !isempty(cont_dims); x_full[cont_dims] = x_cont; end
             x_full[disc_dims] .= d_vals
             return -acq_mean(x_full) # Minimera negativt mean = Maximera mean
        end

        if !isempty(cont_dims)
            n_cont = length(cont_dims)
            starts = [-ones(n_cont) .+ (ones(n_cont) .- -ones(n_cont)) .* ((i .+ 0.5) ./ n_starts) for i in 0:(n_starts - 1)]
            
            for i in 1:n_starts
                x0 = starts[i]
                res = optimize(sub_mean, -1.0 * ones(n_cont), 1.0 * ones(n_cont), x0, Fminbox(LBFGS()); autodiff = :forward)
                
                curr_val = -Optim.minimum(res) # Invertera tillbaka
                if curr_val > best_val
                    best_val = curr_val
                    best_x_full[cont_dims] = Optim.minimizer(res)
                    best_x_full[disc_dims] .= d_vals
                end
            end
        else
            # Bara diskreta
            val = acq_mean(float([d_vals...])) # predict_f vill ha float-vektor även om det är heltal
            if val > best_val
                best_val = val
                best_x_full[disc_dims] .= d_vals
            end
        end
    end
    
    return (fX_max = best_val, X_max = best_x_full)
end