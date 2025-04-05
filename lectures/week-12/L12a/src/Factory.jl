function build(modeltype::Type{MyLayerModel}, data::NamedTuple)::MyLayerModel
    
    # get stuff from data -
    n = data.n
    m = data.m
    σ = data.σ

    # Create a new instance of MyLayerModel
    model = modeltype()
    
    # Initialize the model with data
    model.n = n; # number of inputs
    model.m = m; # number of outputs (nodes in layer)
    model.W = 0.1*randn(m, n+1)  # Random weights for example
    model.σ = σ  # Assign the activation function
    
    # return the model -
    return model
end