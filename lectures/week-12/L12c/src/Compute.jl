

function _evaluate(model::MyLayerModel, x::Array{<: Number})
    
    # initilize -
    m = model.m # number of outputs
    σ = model.σ # activation function
    W = model.W # weights (m x n) includes bias terms
    
    
    # create the output vector -
    y = zeros(m) # output vector of size m
    for i ∈ 1:m
        zᵢ = dot(W[i, :], x) # compute the weighted sum of inputs
        y[i] = σ(zᵢ) # apply the activation function
    end

    return y # return the output vector
end


(model::MyLayerModel)(x::Array{<: Number}) = _evaluate(model, x) # call the function