
function _loss(p::Array{Float64,1}, model::MyElamanRecurrentLayerModel, x::Array{<: Number,1}, y::Array{<: Number,1}, hₒ::Array{<: Number,1})::Float64

    # initialize -
    m = model.number_of_outputs # number of outputs
    h = model.number_of_hidden_units # number of hidden units
    n = model.number_of_inputs # number of inputs
    number_of_time_steps = length(x) # number of time steps
    Y = zeros(number_of_time_steps, number_of_outputs); # output vector

    # we need to reshape the parameters -
    W₁ = p[1:h*n] |> reshape(h, n); # input weights (h x n) = Whx
    W₂ = p[h*n+1:h*n+h*h] |> reshape(h, h); # hidden state weights (h x h) = Whh
    W₃ = p[h*n+h*h+1:h*n+h*h+m*h] |> reshape(m, h); # output weights (m x h) = Why
    b₁ = p[h*n+h*h+m*h+1:h*n+h*h+m*h+h]; # hidden state bias (h) = bh
    b₂ = p[h*n+h*h+m*h+h+1:h*n+h*h+m*h+h+m]; # output bias (m) = by

    # update the model parameters -
    model.Wxh = W₁; # input weights (h x n) = Whx
    model.Whh = W₂; # hidden state weights (h x h) = Whh
    model.Why = W₃; # output weights (m x h) = Why
    model.bh = b₁; # hidden state bias (h) = bh
    model.by = b₂; # output bias (m) = by

     # run the model -
     for i ∈ 1:number_of_time_steps
        
        xᵢ = x[i];  # get the input at time step t
        yᵢ, hᵢ = elmanmodel([xᵢ], hₒ); # run the model

        # store the input and output
        Y[t, :] .= yᵢ;

        # update the hidden state and go around again
        hₒ = hᵢ;
    end
    
    # compute the loss -


end



function learn(model::MyElamanRecurrentLayerModel, x::Array{<: Number,1}, y::Array{<: Number,1}, hₒ::Array{<: Number,1})::MyElamanRecurrentLayerModel

    # initialize -
    loss(p) = _loss(p, model, x, y, hₒ) # loss function
    
    # build the initial guess 
    v₁ = model.Wxh |> vec;
    v₂ = model.Whh |> vec;
    v₃ = model.Why |> vec;
    b₁ = model.bh; # bias for hidden state
    b₂ = model.by; # bias for output
    
    # concatenate the parameters -
    pₒ = [v₁; v₂; v₃; b₁; b₂] # initial guess for the parameters

    # call the optimizer -
    opt_result = Optim.optimize(loss, pₒ, Fminbox(NelderMead()));

end