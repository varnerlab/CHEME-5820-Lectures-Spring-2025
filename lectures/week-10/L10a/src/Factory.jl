function build(modeltype::Type{MySimpleBoltzmannMachineModel}, data::NamedTuple)::MySimpleBoltzmannMachineModel
    
    # Create a new instance of the model
    model = modeltype()
    
    # Initialize the model with the data
    model.W = data.W
    model.b = data.b
    
    # return the model with parameters
    return model
end