function _cbow_loss_function(p, model::MyCBOWEmbeddingModel, x::OneHotVector{UInt32}, y::OneHotVector{UInt32})
    
    # Unpack the weights -
    h = model.h;
    N = model.din;

    # reshape the weights -
    W₁ = p[1:h*N] |> v-> reshape(v, (h, N));
    W₂ = p[(N*h + 1):end] |> v-> reshape(v, (N, h));
    
    # Forward pass -
    h = W₁ * x; # 1. Compute the hidden layer;
    u = W₂ * h; # 2. Compute the output layer
    p = NNlib.softmax(u); # 3. Apply softmax to get probabilities
    
    # Compute the loss -
    loss = -sum(y .* log.(p)); # Cross-entropy loss
    
    # return the loss
    return loss
end


function solve(model::MyCBOWEmbeddingModel, x::OneHotVector{UInt32})
    
    # Forward pass
    h = model.W₁ * x # 1. Compute the hidden layer;
    u = model.W₂ * h  # 2. Compute the output layer
    p = NNlib.softmax(u)  # 3. Apply softmax to get probabilities
    
    # return the probabilities -
    return p
end

function learn!(model::MyCBOWEmbeddingModel, x::OneHotVector{UInt32}, y::OneHotVector{UInt32}; 
    method = Optim.BFGS())::MyCBOWEmbeddingModel
    
    # Initialize -
    h = model.h;
    N = model.din;
    w₁ = model.W₁ |> vec; # flatten the weights
    w₂ = model.W₂ |> vec; # flatten the weights
    p = vcat(w₁, w₂); # concatenate the weights
   
    # setup the objective function -
    loss(p) = _cbow_loss_function(p, model, x, y);
 
    # call the optimizer -
    opt_result = Optim.optimize(loss, p, method);

    θ = Optim.minimizer(opt_result); # get the optimal weights
    model.W₁ = θ[1:h*N] |> v-> reshape(v, (h, N));
    model.W₂ = θ[(N*h + 1):end] |> v-> reshape(v, (N, h));
    
    # return the result -
    return model;
end