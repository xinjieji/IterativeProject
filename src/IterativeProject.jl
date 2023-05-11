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

include("BlackBoxInfo.jl")
export Grid, level_set, ψ, ω, set_field, plot_field, vector_to_field, read_matrices, A

include("DirectSolver.jl")
export direct_solver

include("FastFFT.jl")
export get_diag_2D, inverse_Laplacian

include("IterativeMethods.jl")
export GMRES, GMRES_FFT, BICGStab, BICGStab_FFT

end # module IterativeProject
