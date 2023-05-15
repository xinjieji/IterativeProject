# IterativeProject

This code aims to solve the linear system:

$(I_A + AL^{-1}E_A)\gamma = AL^{-1}f + B\psi_b,$

 $\psi = L^{-1}(f - E_A \gamma),
$

where $L$ is a finite difference discretization matrix for a 2D free space with Dirichlet BC, which can be fastly solved by FFT method. The unknowns are $\gamma$ and $\psi$.

Inside the code, the fast FFT transform for $L$, the GMRES method and the BICGStab($l$) method are implemented.

## Structure

`Grid32\` includes the matrices for testing.

`src\` includes

- The functions to read, transform and plot matrices from the black box in `BlackBoxInfo.jl`.
- A simple backslash solver in `DirectSolver.jl`
- The fast FFT transform functions in `FastFFT.jl`
- The GMRES and BICGStab(l) methods in `IterativeMethods.jl`
- A wrap of all the functions and iterative-FFT combined method in `IterativeProject.jl`

`\test` includes all the test files for these methods. `\test\test.jl` shows some examples to use the package.

## Guide

To activate the package, in Julia,

```julia
] activate /path/to/the/package
```

To use the package, in Julia,

```julia
using IterativeProject
```

Given $Ax = b$, to solve x, we could use the GMRES method:

```julia
x₀ = zeros(b)
x_GMRES, history = GMRES(x₀, A, b)
```

or the BICGStab(l) method:

```julia
l = 1 
x₀ = zeros(b)
x_BICG, history = BICGStab(x₀, A, b, l)
```

For the problem in code, we could construct and plot field while reading the corresponding matrices:

```julia
N = 32 # grid size
grid = Grid((N, N))

# The exact solution
field = set_field(grid, ψ, level_set)
plot_field(field, grid)

# Read matrices from txt files
A, L, E_A, f, B, x_b = read_matrices(N)
```

Then the iterative-FFT combined method can be applied by

```julia
diag_L = get_diag_2D(grid) # pre-computed matrix for fast FFT
x_FFT, history = iteraive_FFT(A, E_A, f, B, x_b, diag_L, grid, "BICGStab", l = 1)
```
