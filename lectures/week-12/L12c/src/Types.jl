abstract type MyAbstractLayerModel end

mutable struct MyFeedforwardLayerModel <: MyAbstractLayerModel
    
    # initilize -
    n::Int # number of inputs
    m::Int # number of outputs
    W::Array{Float64, 2} # weights (m x n) includes bias terms
    σ::Function # activation function

    # constructor -
    MyFeedforwardLayerModel() = new(); # empty constructor
end

mutable struct MyElamanRecurrentLayerModel <: MyAbstractLayerModel
    
    # initilize -
    number_of_inputs::Int # number of inputs
    number_of_outputs::Int # number of outputs
    number_of_hidden_units::Int # number of hidden units
    Whh::Array{Float64, 2} # hidden state weights (h x h)
    Wxh::Array{Float64, 2} # input weights (h x n)
    Why::Array{Float64, 2} # output weights (m x h)
    bh::Array{Float64, 1} # hidden state bias (h)
    by::Array{Float64, 1} # output bias (m)
    σ₁::Function # activation function for hidden state
    σ₂::Function # activation function for output

    # constructor -
    MyElamanRecurrentLayerModel() = new(); # empty constructor
end

mutable struct MyJordanRecurrentLayerModel <: MyAbstractLayerModel
    
    # initilize -
    number_of_inputs::Int # number of inputs
    number_of_outputs::Int # number of outputs
    number_of_hidden_units::Int # number of hidden units
    Whh::Array{Float64, 2} # hidden state weights (h x h)
    Wxh::Array{Float64, 2} # input weights (h x n)
    Why::Array{Float64, 2} # output weights (m x h)
    bh::Array{Float64, 1} # hidden state bias (h)
    by::Array{Float64, 1} # output bias (m)
    σ₁::Function # activation function for hidden state
    σ₂::Function # activation function for output

    # constructor -
    MyJordanRecurrentLayerModel() = new(); # empty constructor
end