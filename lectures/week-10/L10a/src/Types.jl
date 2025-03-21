abstract type AbstractBoltzmannMachineModel end

mutable struct MySimpleBoltzmannMachineModel <: AbstractBoltzmannMachineModel
    
    # fields
    W::Array{Float64,2}; # weight matrix
    b::Vector{Float64}; # bias vector

    # constructor
    MySimpleBoltzmannMachineModel() = new();
end