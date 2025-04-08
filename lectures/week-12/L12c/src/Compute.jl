
function _evaluate(model::MyFeedforwardLayerModel, x::Array{<: Number})
    
    # initilize -
    m = model.m # number of outputs
    σ = model.σ # activation function
    W = model.W # weights (m x n) includes bias terms
    
    # augment the input vector -
    x̂ = [x ; 1.0] # augment the input vector with a bias term
    
    # create the output vector -
    z = zeros(m) # output vector of size m
    for i ∈ 1:m
        z[i] = dot(W[i, :], x̂) |> zᵢ -> σ(zᵢ) # compute the weighted sum of inputs
    end

    return z # return the output vector
end

function _evaluate(model::MyElamanRecurrentLayerModel, x::Array{<: Number}, h::Array{<: Number})
    
    # initilize -
    number_of_time_steps = size(x,1) # number of time steps
    m = model.dout # number of outputs
    h = model.dh # number of hidden units

end


(model::MyFeedforwardLayerModel)(x::Array{<: Number,1}) = _evaluate(model, x) # call the function
(model::MyElamanRecurrentLayerModel)(x::Array{<: Number}, h::Array{<: Number}) = _evaluate(model, x, h) # call the function