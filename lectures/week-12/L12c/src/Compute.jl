
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
    σ₁ = model.σ₁ # activation function for hidden state
    σ₂ = model.σ₂ # activation function for output
    Wxh = model.Wxh # input weights (h x n)
    Whh = model.Whh # hidden state weights (h x h)
    Why = model.Why # output weights (m x h)
    bh = model.bh # hidden state bias (h)
    by = model.by # output bias (m)

    # compute the new hidden state -
    h̄ = σ₁.(Wxh * x .+ Whh * h .+ bh) # element-wise activation

    # compute the output -
    y = σ₂.(Why * h̄ .+ by) # element-wise activation

    # return -
    return y, h̄ # return the output and new hidden state
end


(model::MyFeedforwardLayerModel)(x::Array{<: Number,1}) = _evaluate(model, x) # call the function
(model::MyElamanRecurrentLayerModel)(x::Array{<: Number,1}, h::Array{<: Number,1}) = _evaluate(model, x, h) # call the function