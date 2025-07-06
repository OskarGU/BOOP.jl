# Here I make a first dummy function to get the documentation online.


"""
```
    first_func(x)
```
    
A first dummy function that returns the log of the square of the input value.

    ```julia-repl
julia> T = 500; ϕ = [sin.(2π*(1:T)/T) -0.1*ones(T)]; θ = zeros(T); σ = 1; μ = 2;
julia> y = simTvARMA(ϕ, θ, μ, σ);
julia>  tvPeriodogram(y, 25, 15)
```
"""
function first_func(x)
    return log(x^2)
end


X_train = [1.0, 2.5, 4.0];  y_train = [sin(x) for x in X_train];
gp_model = GP(X_train', y_train, MeanZero(), SE(0.0, 0.0));
optimize!(gp_model);
# 2. Define the best observed value and a candidate point
y_best = minimum(y_train);
x_candidate = 3.0;
# 3. Compute Expected Improvement
ei = expected_improvement(gp_model, x_candidate, y_best; ξ=0.01)