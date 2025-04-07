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
    σ₁ = data.σ₁ # activation function for hidden state
    σ₂ = data.σ₂ # activation function for output

    # Create a new instance of MyLayerModel
    model = modeltype()
    
    # Initialize the model with data
    model.number_of_inputs = n; # number of inputs
    model.number_of_outputs = m; # number of outputs (nodes in layer)
    model.number_of_hidden_units = h; # number of hidden units
    model.Whh = 0.1*randn(h, h)  # Random weights for example
    model.Wxh = 0.1*randn(h, n)  # Random weights for example
    model.Why = 0.1*randn(m, h)  # Random weights for example
    model.bh = 0.1*randn(h)      # Random bias for example
    model.by = 0.1*randn(m)      # Random bias for example
    model.σ₁ = σ₁  # Assign the activation function for hidden state
    model.σ₂ = σ₂  # Assign the activation function for output
    
    # return the model -
    return model
end