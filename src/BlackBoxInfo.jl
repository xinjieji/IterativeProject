# This file help visualize and transform the matrix information

"""
Grid Information
"""
struct Grid
    dims::Tuple{Int64, Int64} # resolution of the grid
end

# return the position of an index
function pos(grid::Grid, i)
    return Float64[(i[1] - 1)/grid.dims[1], (i[2] - 1)/grid.dims[2]]
end

# return the grid spacing
function grid_spacing(grid::Grid)
    return [1/grid.dims[1], 1/grid.dims[2]]
end

"""
Interface Information
"""
# parameters of the star
star_center = [0.51,0.52]
average_radius = 0.25
deviation_radius = 0.01
number_points = 7

function level_set(x)
    p = PolarFromCartesian()(x - star_center)
    return p.r - (average_radius + deviation_radius*cos(number_points*p.θ))
end

"""
Exact field and generate field
"""
# exact solutions for domain [0, 1], ρ_f = 1.0 by default
ψ(x) = -cos(x[1]*2*pi)*cos(x[2]*2*pi)
ω(x) = -8*pi^2*cos(x[1]*2*pi)*cos(x[2]*2*pi)

# store value on the grid
function set_field(grid::Grid, func::Function, levelset::Function = (x) -> 1.0)
    field = CircularArray(0.0, grid.dims)
    for I in eachindex(field)
        x = pos(grid, Tuple(I))
        if levelset(x) > 0.0
            field[I] = func(x)
        end
    end
    return field
end

"""
Plotting
"""
# flip the field for accurate plot
pyplot_flip(field::AbstractArray) = reverse(field', dims=1)
# extent the filed for accurate plot
function pyplot_extent(grid::Grid) 
    h = grid_spacing(grid)
    return (-h[1]/2, 1 - h[1]/2, -h[2]/2, 1 - h[2]/2)
end

function plot_field(field, grid::Grid; kwargs...)
    fig, ax = subplots()
    imsh = ax.imshow(pyplot_flip(field), extent = pyplot_extent(grid), cmap = "coolwarm"; kwargs...)
    fig.colorbar(imsh)
    return fig
end


"""
Vector and field transformation
"""
# transfer a field vector back to a matrix
function vector_to_field(vec, grid::Grid)
    sz = grid.dims
    field = CircularArray(0.0, sz)
    n = 1
    for i in 1:sz[1]
        for j in 1:sz[2]
            field[i,j] = vec[n]
            n += 1
        end
    end
    return field
end

"""
Read matrices from txt files based on grid size
> A, L, E_A, f, B, x_b = read_matrices(32)
"""

function read_matrices(N)
    file1 = open("Grid$N/A.txt", "r")
    A = sparse(readdlm(file1))
    close(file1)

    file2 = open("Grid$N/L.txt", "r")
    L = sparse(readdlm(file2))
    close(file2)

    file3 = open("Grid$N/E_A.txt", "r")
    E_A = sparse(readdlm(file3))
    close(file3)

    file4 = open("Grid$N/f.txt", "r")
    f = readdlm(file4)
    close(file4)

    file5 = open("Grid$N/B.txt", "r")
    B = sparse(readdlm(file5))
    close(file5)

    file6 = open("Grid$N/bc.txt", "r")
    x_b = readdlm(file6)
    close(file6)

    return A, L, E_A, f, B, x_b
end

"""
Function to represent the left-hand-side of the linear system
(I_A + AL^{-1}E_A)*γ
"""
function A_func(x, A, E_A, diag_L, grid::Grid)
    return x + A*inverse_Laplacian(reshape(E_A*x, grid.dims), diag_L)
end
