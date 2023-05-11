# This file stores all the iterative methods for Ax = b
# or Iterative-FFT problems

"""
GMRES
"""
# Reference form https://tobydriscoll.net/fnc-julia/krylov/gmres.html
function GMRES(x₀, A, b; iter = 1000, tol = 1e-13)
    x = x₀
    n = length(b)
    Q = zeros((n, iter+1))
    Q[:,1] = b/norm(b)
    H = zeros((iter+1, iter))
    residual = [norm(b)]
    for j = 1:iter
        # Next step of Arnoldi process
        v = A*Q[:,j]
        for i = 1:j
            H[i,j] = dot(Q[:,i], v)
            v = v - H[i,j]*Q[:,i]
        end
        H[j+1,j] = norm(v)
        Q[:,j+1] = v/H[j+1,j]

        # Solve least squares problem
        r = vcat(norm(b), zeros(j))
        z = H[1:j+1,1:j] \ r
        x = Q[:,1:j]*z
        residual = vcat(residual, norm(b - A*x))
        if norm(b - A*x) < tol
            break
        end
    end
    return x, residual
end

function GMRES_FFT(x₀, A::Function, b; iter = 1000, tol = 1e-13)
    x = x₀
    n = length(b)
    Q = zeros((n, iter+1))
    Q[:,1] = b/norm(b)
    H = zeros((iter+1, iter))
    residual = [norm(b)]
    for j = 1:iter
        # Next step of Arnoldi process
        v = A(Q[:,j])
        for i = 1:j
            H[i,j] = dot(Q[:,i], v)
            v = v - H[i,j]*Q[:,i]
        end
        H[j+1,j] = norm(v)
        Q[:,j+1] = v/H[j+1,j]

        # Solve least squares problem
        r = vcat(norm(b), zeros(j))
        z = H[1:j+1,1:j] \ r
        x = Q[:,1:j]*z
        tmp = norm(b - A(x))
        residual = vcat(residual, tmp)
        if tmp < tol
            break
        end
    end
    return x, residual
end

"""
BICGStab(l)
"""
# Reference from: 
# BICGSTAB(L) FOR LINEAR EQUATIONS INVOLVING UNSYMMETRIC MATRICES WITH COMPLEX SPECTRUM
function BICGStab(x₀, A, b, l; iter = 1000, tol = 1e-13)
    r̃₀ = rand(length(b))
    û = zeros(length(b), l+1)
    r̂ = zeros(length(b), l+1)
    u = zeros(length(b))
    r = b - A*x₀
    x = x₀
    residual = [norm(r)]
    ρ = [1.0, 1.0]
    α = 0.0
    ω = 1.0
    # iteration
    for n = 1:iter
        û[:,1] = u; r̂[:,1] = r; x̂ = x
        ρ[1] = -ρ[1]*ω

        # BI-CG part
        for j = 1:l
            ρ[2] = dot(r̂[:,j], r̃₀); β = α*ρ[2]/ρ[1]; ρ[1] = ρ[2]
            for i = 1:j
                û[:,i] = r̂[:,i] - β*û[:,i]
            end
            û[:,j+1] = A*û[:,j]
            γ = dot(û[:,j+1], r̃₀); α = ρ[1]/γ
            for i = 1:j
                r̂[:,i] = r̂[:,i] - α*û[:,i+1]
            end
            r̂[:,j+1] = A*r̂[:,j]; x̂ = x̂ + α*û[:,1]
        end
        # MR part
        σ = zeros(l+1)
        γ′ = zeros(l+1)
        τ = zeros(l,l+1)
        for j = 2:l+1
            if j != 2
                for i = 2:j-1
                    τ[i,j] = dot(r̂[:,j], r̂[:,i])/σ[i]
                    r̂[:,j] = r̂[:,j] - τ[i,j]*r̂[:,i]
                end
            end
            σ[j] = dot(r̂[:,j], r̂[:,j]); γ′[j] = 1/σ[j]*dot(r̂[:,1], r̂[:,j])
        end
        γ = copy(γ′); γ′′ = copy(γ); ω = γ[l+1]
        if l >= 2
            for j = l:-1:2
                γ[j] = γ′[j] - dot(τ[j,j+1:l+1], γ[j+1:l+1])
            end
            for j = 2:l
                γ′′[j] = γ[j+1] + dot(τ[j,j+1:l], γ[j+2:l+1])
            end
        end
        # update
        x̂  = x̂ + γ[2]*r̂[:,1]; r̂[:,1] = r̂[:,1] - γ′[l+1]*r̂[:,l+1]; û[:,1] = û[:,1] - γ[l+1]*û[:,l+1]
        if l >= 2
            for j = 2:l
                û[:,1] = û[:,1] - γ[j]*û[:,j]; x̂ = x̂ + γ′′[j]*r̂[:,j]; r̂[:,1] = r̂[:,1] - γ′[j]*r̂[:,j]
            end
        end
        u = û[:,1]; x = x̂; r = r̂[:,1]
        residual = vcat(residual, norm(r))
        if norm(r) < tol
            break
        end
    end
    return x, residual
end

function BICGStab_FFT(x₀, A::Function, b, l; iter = 1000, tol = 1e-13)
    r̃₀ = rand(length(b))
    û = zeros(length(b), l+1)
    r̂ = zeros(length(b), l+1)
    u = zeros(length(b))
    r = b - A(x₀)
    x = x₀
    residual = [norm(r)]
    ρ = [1.0, 1.0]
    α = 0.0
    ω = 1.0
    # iteration
    for n = 1:iter
        û[:,1] = u; r̂[:,1] = r; x̂ = x
        ρ[1] = -ρ[1]*ω

        # BI-CG part
        for j = 1:l
            ρ[2] = dot(r̂[:,j], r̃₀); β = α*ρ[2]/ρ[1]; ρ[1] = ρ[2]
            for i = 1:j
                û[:,i] = r̂[:,i] - β*û[:,i]
            end
            û[:,j+1] = A(û[:,j])
            γ = dot(û[:,j+1], r̃₀); α = ρ[1]/γ
            for i = 1:j
                r̂[:,i] = r̂[:,i] - α*û[:,i+1]
            end
            r̂[:,j+1] = A(r̂[:,j]); x̂ = x̂ + α*û[:,1]
        end
        # MR part
        σ = zeros(l+1)
        γ′ = zeros(l+1)
        τ = zeros(l,l+1)
        for j = 2:l+1
            if j != 2
                for i = 2:j-1
                    τ[i,j] = dot(r̂[:,j], r̂[:,i])/σ[i]
                    r̂[:,j] = r̂[:,j] - τ[i,j]*r̂[:,i]
                end
            end
            σ[j] = dot(r̂[:,j], r̂[:,j]); γ′[j] = 1/σ[j]*dot(r̂[:,1], r̂[:,j])
        end
        γ = copy(γ′); γ′′ = copy(γ); ω = γ[l+1]
        if l >= 2
            for j = l:-1:2
                γ[j] = γ′[j] - dot(τ[j,j+1:l+1], γ[j+1:l+1])
            end
            for j = 2:l
                γ′′[j] = γ[j+1] + dot(τ[j,j+1:l], γ[j+2:l+1])
            end
        end
        # update
        x̂  = x̂ + γ[2]*r̂[:,1]; r̂[:,1] = r̂[:,1] - γ′[l+1]*r̂[:,l+1]; û[:,1] = û[:,1] - γ[l+1]*û[:,l+1]
        if l >= 2
            for j = 2:l
                û[:,1] = û[:,1] - γ[j]*û[:,j]; x̂ = x̂ + γ′′[j]*r̂[:,j]; r̂[:,1] = r̂[:,1] - γ′[j]*r̂[:,j]
            end
        end
        u = û[:,1]; x = x̂; r = r̂[:,1]
        residual = vcat(residual, norm(r))
        if norm(r) < tol
            break
        end
    end
    return x, residual
end

