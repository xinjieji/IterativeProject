# This file include FastFFT method for solving a L^{-1} on unifrom grid

# calculate eigenvalues of 1D circulant FD matrix (k start from 1)
function get_eigenvalues(N,fd_order)
    # create the first column
    indices, fd_stencil = centered_second_derivative_stencil_dict_[fd_order]
    s = OffsetArray(fd_stencil, indices)
    c1 = zeros(N)
    for n in indices
        if n >= 0
            c1[n+1] = s[n]
        else
            c1[N+1+n] = s[n]
        end
    end

    # calculate the eigenvalues of it
    λ_set = zeros(N)
    λ_set = Complex.(λ_set)
    for k in 1:N
        λ = 0.0
        for n in 1:N
            λ += c1[n]*exp(-2*pi*im*(k-1)*(n-1)/N)
        end
        λ_set[k] = λ
    end
    return λ_set
end

function get_diag_2D(grid::Grid; fd_order = 2)
    sz = grid.dims
    h = grid_spacing(grid)
    eigen_1D_x = get_eigenvalues(sz[1],fd_order)/h[1]^2
    eigen_1D_y = get_eigenvalues(sz[2],fd_order)/h[2]^2
    diag_1D_x = diagm(eigen_1D_x)
    diag_1D_y = diagm(eigen_1D_y)
    diag_RC = kron(diag_1D_x,1*Matrix(I, sz[1], sz[1])) + kron(1*Matrix(I, sz[1], sz[1]), diag_1D_y)
    diag_vector = diag(diag_RC)
    diag_vector[1] = 1.0
    return diag_vector
end

"""
Example: given grid , output f, and FD order(in this project, 2)
> diag_L = get_diag_2D(grid)
> x = inverse_Laplacian(reshape(f,grid.dims), diag_L)
"""

function inverse_Laplacian(vector, diag_RC)
    fft_vector = vec(fft(vector))
    fft_solution = fft_vector./diag_RC
    fft_solution[1] = 0
    solution = ifft(reshape(fft_solution, size(vector)))
    return real(vec(solution))
end
