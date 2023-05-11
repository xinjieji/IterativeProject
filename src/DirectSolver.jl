# This file is a direct solver
# The idea is directly solve the linear system (L + E_A*A)x = f - E_A*B*x_b by backslash

function direct_solver(A, L, E_A, f, B, x_b)
    LHS = L + E_A*A
    RHS = f - E_A*B*x_b
    return LHS\RHS
end