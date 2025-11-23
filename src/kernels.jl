# ==============================================================================
# Implement GARRIDO-MERCHÁN KERNEL for GPs
# ==============================================================================

# Our new kerel use a standard one and just make sure the kernel is constant between the discrete points.
struct GarridoMerchanKernel{K<:Kernel} <: Kernel
    base_kernel::K
    integer_dims::Vector{Int} # Which dimension of X that are discrete.
    integer_ranges::Vector{UnitRange{Int}} # Range of allowed values.
end

# We need to update these functions to work with our new kernel.
import GaussianProcesses: cov, get_params, set_params!, num_params, get_priors, dKij_dθp, set_priors!

# Here is the main trick, we round the data within the kernel to tell it that 2.1=2.3=2.0, only
# difference in these points are measurement noise.
function GaussianProcesses.cov(k::GarridoMerchanKernel, x::AbstractVector, y::AbstractVector)
    x_snapped = copy(x)
    y_snapped = copy(y)
    
    # Just rounding
    for dim in k.integer_dims
        x_snapped[dim] = round(x_snapped[dim])
        y_snapped[dim] = round(y_snapped[dim])
    end
    return cov(k.base_kernel, x_snapped, y_snapped)
end

# --- Update some paackage functions to work with the new kernel ---
GaussianProcesses.get_params(k::GarridoMerchanKernel) = get_params(k.base_kernel)
GaussianProcesses.num_params(k::GarridoMerchanKernel) = num_params(k.base_kernel)
GaussianProcesses.get_priors(k::GarridoMerchanKernel) = get_priors(k.base_kernel)

function GaussianProcesses.set_params!(k::GarridoMerchanKernel, params::AbstractVector)
    set_params!(k.base_kernel, params)
end

function GaussianProcesses.set_priors!(k::GarridoMerchanKernel, priors::Array)
    set_priors!(k.base_kernel, priors)
end

# Import function KernelData
#using GaussianProcesses: KernelData

# This is for the optimizer to be able to find the gradients w.r.t. the GP parameters.
function GaussianProcesses.dKij_dθp(k::GarridoMerchanKernel, X::AbstractMatrix, Y::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int)
    xi = X[:, i]
    yj = Y[:, j]
    
    # We need to round the data before passing it to the base kernel.
    for d in k.integer_dims
        xi[d] = round(xi[d])
        yj[d] = round(yj[d])
    end
    
    # Temporary matrices as GaussianProcesses.jl demand as input to dKij_dθp
    X_snap = reshape(xi, :, 1)
    Y_snap = reshape(yj, :, 1)
    
    # Derivative between the rounded points.
    return dKij_dθp(k.base_kernel, X_snap, Y_snap, 1, 1, p, dim)
end