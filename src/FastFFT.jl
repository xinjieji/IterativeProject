# This file include FastFFT method for solving a L^{-1} on unifrom grid

function get_diag_2D(grid::Grid; fd_order = 2)
    sz = grid.dims
    h = grid_spacing(grid)
    diag_1D = zeros(sz[1])
    diag_identity = ones(sz[1])
    if fd_order == 2
        for i in 1:sz[1]
            diag_1D[i] = -2 + 2*cos(pi*i/(sz[1]+1))
        end
    # 4th order does not convergence
    elseif fd_order == 4
        for i in 1:sz[1]
            diag_1D[i] = -1/3*(cos(pi*i/(sz[1]+1)) -1)*(cos(pi*i/(sz[1]+1)) -7)
        end
    end
    A = Diagonal(diag_1D)
    I_matrix = Diagonal(diag_identity)
    c = kron(A,I_matrix) + kron(I_matrix, A)
    return diag(c)/h[1]^2
end

"""
Example: given grid , output f, and FD order(in this project, 2)
> diag_L = get_diag_2D(grid)
> x = inverse_Laplacian(f, diag_L)
"""

function inverse_Laplacian(vector, diag_RC)
    fft_vector = vec(FFTW.r2r(vector, FFTW.RODFT00))
    fft_solution = fft_vector./diag_RC
    solution = FFTW.r2r(reshape(fft_solution, size(vector)), FFTW.RODFT00)/(4*(size(vector)[1]+1)^2)
    return vec(solution)
end
