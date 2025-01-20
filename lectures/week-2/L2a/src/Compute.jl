function _projection(v::Array{Float64,1}, u::Array{Float64,1})::Array{Float64,1}
    return dot(v, u) * u;
end

function _orthogonalize(A::Array{<:Number,2}, algorithm::ClassicalGramSchmidtAlgorithm)::Array{Float64,2}
    
    # initialize
    number_of_rows = size(A, 1); # how many rows do we have?
    number_of_cols = size(A, 2); # how many columns do we have?
    Q = zeros(Float64, number_of_rows, number_of_cols); # initialize the Q matrix

    # we are going to find the orthogonal basis for the columns of A
    for i = 1:number_of_cols
        v = A[:, i]; # get the i-th column
        for j = 1:i-1 # all
            q = Q[:, j]; # get the j-th column
            v = v - _projection(v, q); # subtract the projection
        end
        Q[:, i] = v / norm(v); # normalize the vector, q₁ = v₁ / ||v₁||
    end
    return Q;
end

function _orthogonalize(A::Array{<:Number,2}, algorithm::ModifiedGramSchmidtAlgorithm)::Array{Float64,2}

    # initialize
    number_of_rows = size(A, 1); # how many rows do we have?
    number_of_cols = size(A, 2); # how many columns do we have?
    
    # copy the matrix A -
    V = copy(A);

    for j ∈ 1:number_of_cols
        q = V[:, j]/norm(V[:,j]) # get the j-th column
        for i ∈ j+1:number_of_cols
            V[:, i] = V[:, i] - dot(V[:, i], q) * q
        end
        Q[:, j] = q; # capture the orthogonal vector
    end
    return Q;
end

"""
    poweriteration(A::Array{<:Number,2}, v::Array{<:Number,1}; 
        maxiter::Int = 100, ϵ::Float64 = 0.0001)

This function computes the dominant eigenvector and eigenvalue of a matrix using the power iteration method.

### Arguments
- `A::Array{<:Number,2}`: A square matrix of real numbers.
- `v::Array{<:Number,1}`: An initial guess for the eigenvector.
- `maxiter::Int = 100`: The maximum number of iterations (optional).
- `ϵ::Float64 = 0.0001`: The convergence criterion (optional).

### Output
- A tuple containing the dominant eigenvector and eigenvalue.
"""
function poweriteration(A::Array{<:Number,2}, v::Array{<:Number,1}; 
    maxiter::Int = 100, ϵ::Float64 = 0.0001):: Tuple{Array{<:Number,1}, Number}

    # initialize
    loopcount = 1;
    should_we_stop = false;

    while (should_we_stop == false)
        
        # compute the next iteration
        w = A * v;
        w = w / norm(w);

        # check if we should stop
        if (norm(w - v) ≤ ϵ || loopcount ≥ maxiter)
            should_we_stop = true;
            println("Converged in $(loopcount) iterations"); # let the user know how many iterations it took
        else
            v = w; # update the vector
            loopcount = loopcount + 1; # update the loop count
        end
    end
    
    # compute the eigenvalue -
    λ = dot(A * v, v) / dot(v, v);

    # return the result
    return (v, λ);
end

"""
    function orthogonalize(A::Array{<:Number,2}, algorithm::T)::Array{Float64,2} where {T<:AbstractGramSchmidtAlgorithm}

This function computes the orthogonal basis of a matrix using the specified algorithm, which can be either `ClassicalGramSchmidtAlgorithm` or `ModifiedGramSchmidtAlgorithm`.

### Arguments
- `A::Array{<:Number,2}`: A matrix of real numbers.
- `algorithm::T`: An instance of the `AbstractGramSchmidtAlgorithm` subtype. This can be either `ClassicalGramSchmidtAlgorithm` or `ModifiedGramSchmidtAlgorithm`.

### Output
- An orthogonal basis for the columns of the matrix `A`.
"""
function orthogonalize(A::Array{<:Number,2}, algorithm::T)::Array{Float64,2} where {T<:AbstractGramSchmidtAlgorithm}
    return _orthogonalize(A, algorithm);
end