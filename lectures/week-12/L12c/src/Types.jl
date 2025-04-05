abstract type MyAbstractLayerModel end

mutable struct MyLayerModel <: MyAbstractLayerModel
    
    # initilize -
    n::Int # number of inputs
    m::Int # number of outputs
    W::Array{Float64, 2} # weights (m x n) includes bias terms
    σ::Function # activation function

    # constructor -
    MyLayerModel() = new(); # empty constructor
end