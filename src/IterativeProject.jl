module IterativeProject

# Solve the system:
# (I_A + AL^{-1}E_A)*γ = AL^{-1}f + Bx_b
# x = L^{-1}(f - E_A*γ)
# In this project, the input is ω, the BC is ψ_b, and the result is ψ

using LinearAlgebra
using CircularArrays
using CoordinateTransformations
using PyPlot
using SparseArrays
using DelimitedFiles
using FFTW
using BenchmarkTools
using OffsetArrays

include("BlackBoxInfo.jl")
export Grid, level_set, ψ, ω, set_field, plot_field, vector_to_field, read_matrices, A_func

include("DirectSolver.jl")
export direct_solver

include("FastFFT.jl")
export get_diag_2D, inverse_Laplacian

include("IterativeMethods.jl")
export GMRES, GMRES_FFT, BICGStab, BICGStab_FFT

# Wrap up
# iterative method with FFT
"""
precompute:
> diag_L = get_diag_2D(grid)
"""
function iteraive_FFT(A, E_A, f, B, x_b, diag_L, grid::Grid, method; l = 1, tol = 1/grid.dims[1]^2)
    γ = zeros(size(B,1)); history = []
    rhs = A*inverse_Laplacian(reshape(f,grid.dims), diag_L) + B*x_b
    if method == "GMRES"
        γ, history = GMRES_FFT(γ, (x)->A_func(x,A,E_A,diag_L,grid) , rhs, tol = tol)
    elseif method == "BICGStab"
        γ, history = BICGStab_FFT(γ, (x)->A_func(x,A,E_A,diag_L,grid) , rhs, l, tol = tol)
    else
        throw("Please choose a method from GMRES and BICGStab")
    end
    x = inverse_Laplacian(reshape(f - E_A*γ, grid.dims), diag_L)
    return x, history
end

export iteraive_FFT

end # module IterativeProject
