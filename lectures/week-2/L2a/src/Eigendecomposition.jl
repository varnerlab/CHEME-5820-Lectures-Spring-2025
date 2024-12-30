function _rref!(A::Matrix{T}, ɛ=T <: Union{Rational,Integer} ? 0 : eps(norm(A,Inf))) where T
    nr, nc = size(A)
    i = j = 1
    while i <= nr && j <= nc
        (m, mi) = findmax(abs.(A[i:nr,j]))
        mi = mi+i - 1
        if m <= ɛ
            if ɛ > 0
                A[i:nr,j] .= zero(T)
            end
            j += 1
        else
            for k=j:nc
                A[i, k], A[mi, k] = A[mi, k], A[i, k]
            end
            d = A[i,j]
            for k = j:nc
                A[i,k] /= d
            end
            for k = 1:nr
                if k != i
                    d = A[k,j]
                    for l = j:nc
                        A[k,l] -= d*A[i,l]
                    end
                end
            end
            i += 1
            j += 1
        end
    end
    A
end

function _upper_triangular(matrix::Matrix)
    A = copy(matrix)
    (n, m) = size(A)
    for k = 1:min(n,m)
        for i = k+1:n
            l = A[i,k] / A[k,k]
            for j = k:m
                A[i,j] = A[i,j] - l*A[k,j]
            end
        end
    end
    return A
end

"""
    ⊗(a::Array{Float64,1},b::Array{Float64,1}) -> Array{Float64,2}

Compute the outer product of two vectors `a` and `b` and returns a matrix.

### Arguments
- `a::Array{Float64,1}`: a vector of length `m`.
- `b::Array{Float64,1}`: a vector of length `n`.

### Returns
- `Y::Array{Float64,2}`: a matrix of size `m x n` such that `Y[i,j] = a[i]*b[j]`.
"""
function ⊗(a::Array{Float64,1},b::Array{Float64,1})::Array{Float64,2}

    # initialize -
    m = length(a)
    n = length(b)
    Y = zeros(m,n)

    # main loop 
    for i ∈ 1:m
        for j ∈ 1:n
            Y[i,j] = a[i]*b[j]
        end
    end

    # return 
    return Y
end

"""
    myqriteration(A::Array{Float64,2}; maxiter::Int64 = 10, tolerance::Float64 = 1e-9) -> Tuple

Computes the eigenvalues and eigenvectors of a real matrix `A` using the QR iteration method.

### Arguments
- `A::Array{Float64,2}`: a real matrix of size `n x n`.
- `maxiter::Int64`: the maximum number of iterations (default is `10`).
- `tolerance::Float64`: the tolerance for the stopping criterion (default is `1e-9`).

### Returns
- `Tuple`: a tuple of two elements: the first element is an array of eigenvalues and the second element is a dictionary of eigenvectors.
"""
function myqriteration(A::Array{Float64,2}; 
    maxiter::Int64 = 10, ϵ::Float64 = 1e-9)::Tuple{Array{Float64,1}, Dict{Int64,Array{Float64,1}}}

    # initialize 
    number_of_rows = size(A,1);
    AD = A;
    eigenvectors = Dict{Int64,Array{Float64,1}}()

    # compute iteration zero -
    Fₒ = qr(AD)
    Qₒ = Matrix(Fₒ.Q);
    λ = diag(Qₒ) # get initial guess of eigenvalues

    # Phase 1: compute the eigenvalues
    is_ok_to_stop = false; # phase 1 flag
    loop_counter = 1;
    while (is_ok_to_stop == false)
        
        # factor -
        F = qr(AD)
        Q = Matrix(F.Q);

        # update
        AD = transpose(Q)*AD*Q

        # check: should we continute to iterate?
        λ′ = diag(AD);
        if (norm(λ - λ′) ≤ ϵ || loop_counter ≥ maxiter)
            
            # update the values
            λ = λ′
            is_ok_to_stop = true;
            println("Converged in $(loop_counter) iterations")
        else
            # have not hit the tolerance, and we still have iterations left
            λ = λ′
            loop_counter += 1; # increment the loop counter
        end
    end

    # sort the eigenvalues -
    sort!(λ)

    # Phase 2: compute the eigenvectors
    b = zeros(number_of_rows);
    for i ∈ 1:number_of_rows
        
        # setup the homogenous system -
        AH = A - λ[i]*Matrix{Float64}(1.0*I, number_of_rows, number_of_rows);
        
        b[i] = 1e-10; # perturbation
        EVUS = AH \ b

        # compute a scaling normalizing parameter
        d = norm(EVUS);
    
        # compute the scaled eigenvector - not sure why the (-1)^(i+1)?
        eigenvectors[i] = ((-1)^(i+1))*(1/d)*EVUS
    end

    # return -
    return (λ, eigenvectors)
end