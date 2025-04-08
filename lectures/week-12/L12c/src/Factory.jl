function build(modeltype::Type{MyFeedforwardLayerModel}, data::NamedTuple)::MyFeedforwardLayerModel
    
    # get stuff from data -
    n = data.n
    m = data.m
    σ = data.σ

    # Create a new instance of MyLayerModel
    model = modeltype()
    
    # Initialize the model with data
    model.n = n; # number of inputs
    model.m = m; # number of outputs (nodes in layer)
    model.W = randn(m, n)  # Random weights for example
    model.σ = σ  # Assign the activation function
    
    # return the model -
    return model
end

function build(modeltype::Type{MyElamanRecurrentLayerModel}, data::NamedTuple)::MyElamanRecurrentLayerModel
    
    # get stuff from data -
    n = data.number_of_inputs; # number of inputs
    m = data.number_of_outputs; # number of outputs
    h = data.number_of_hidden_units;
    batchsize = data.batchsize; # batch size
    σ₁ = data.σ₁ # activation function for hidden state
    σ₂ = data.σ₂ # activation function for output

    # Construct and populate the model -
    model = modeltype()
    model.din = n; # number of inputs
    model.dout = m; # number of outputs
    model.dh = h; # number of hidden units
    model.batchsize = batchsize; # batch size
    
    # layer - 
    cell = Flux.RNNCell(n => h, σ₁) # RNN cell
    model.model = Flux.Chain(hidden = cell, output = Flux.Dense(h => m, σ₂)) # RNN layer
    
    # return the model -
    return model
end