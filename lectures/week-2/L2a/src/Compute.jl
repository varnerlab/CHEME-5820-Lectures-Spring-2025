function _projection(v::Array{Float64,1}, u::Array{Float64,1})::Array{Float64,1}
    return (dot(v, u) / dot(u, u)) * u;
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

function orthogonalize(A::Array{<:Number,2}, algorithm::T)::Array{Float64,2} where {T<:AbstractGramSchmidtAlgorithm}

    # initialize
    number_of_rows = size(A, 1); # how many rows do we have?
    number_of_cols = size(A, 2); # how many columns do we have?
    Q = zeros(Float64, number_of_rows, number_of_cols); # initialize the Q matrix

    # we are going to find the orthogonal basis for the columns of A
    for i = 1:number_of_cols
        
        v = A[:, i]; # get the i-th column
        
        for j = 1:i-1
            
            if (isa(algorithm, ClassicalGramSchmidtAlgorithm))
                v = v - _projection(A[:, i], Q[:, j]); # subtract the projection
            elseif (isa(algorithm, ModifiedGramSchmidtAlgorithm))
                v = v - _projection(v, Q[:,j]); # subtract the projection
            end
        end
        Q[:, i] = v / norm(v); # normalize the vector
    end

    return Q;
end