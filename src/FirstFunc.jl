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