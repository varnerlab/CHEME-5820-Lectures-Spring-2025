abstract type AbstractClassificationAlgorithm end

mutable struct MyPerceptronClassificationModel <: AbstractClassificationAlgorithm
    
    # data -
    β::Vector{Float64}; # coefficients
    mistakes::Int64; # number of mistakes that are are willing to make

    # empty constructor -
    MyPerceptronClassificationModel() = new();
end