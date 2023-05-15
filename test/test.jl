using IterativeProject

N = 32 # grid size, 32, 64, 128, 256
grid = Grid((N, N))

# The exact solution
field = set_field(grid, ψ, level_set)
plot_field(field, grid)

# Read matrices from txt files
A, L, E_A, f, B, x_b = read_matrices(N)

# Plot matrix
using PyPlot
fig, ax = subplots()
ax.spy(L, aspect= "auto")
ax.set_title("Sparsity of L")
gcf()

# use backslash to solve the linear system
x = direct_solver(A, L, E_A, f, B, x_b)
x_field = vector_to_field(x, grid)
# Error
err = maximum(abs.(x_field - field))

# Get the LHS and RHS of the linear system
L_hat = L + E_A*A
f_hat = f - E_A*B*x_b

fig, ax = subplots()
ax.spy(L_hat, aspect= "auto")
ax.set_title("Sparsity of L hat")
gcf()

x₀ = zeros(size(f))
# Use BICGStab(1)
x_BICG, history = BICGStab(x₀, L_hat, f_hat, 1, tol = 1/grid.dims[1]^2)
# Use GMRES
x_GMRES, history = GMRES(x₀, L_hat, f_hat, tol = 1/grid.dims[1]^2)

x_BICG = vector_to_field(x_BICG, grid)
# Error
err = maximum(abs.(x_BICG - field))
# Plot
plot_field(x_BICG, grid)

# iterative method with FFT
# (I_A + AL^{-1}E_A)*γ = AL^{-1}f + Bx_b
# x = L^{-1}(f - E_A*γ)
diag_L = get_diag_2D(grid)
x_FFT, history = iteraive_FFT(A, E_A, f, B, x_b, diag_L, grid, "BICGStab", l = 1)
x_FFT = vector_to_field(x_FFT, grid)
err = maximum(abs.(x_FFT - field))
plot_field(x_FFT, grid)