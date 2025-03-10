abstract type AbstractBanditAlgorithmModel end
abstract type AbstractBanditAlgorithmContextModel end


mutable struct MyExploreFirstAlgorithmModel <: AbstractBanditAlgorithmModel

    # data -
    K::Int64 # number of arms

    # constructor -
    MyExploreFirstAlgorithmModel() = new();
end

mutable struct MyEpsilonGreedyAlgorithmModel <: AbstractBanditAlgorithmModel

    # data -
    K::Int64 # number of arms

    # constructor -
    MyEpsilonGreedyAlgorithmModel() = new();
end

mutable struct MyUCB1AlgorithmModel <: AbstractBanditAlgorithmModel

    # data -
    K::Int64 # number of arms

    # constructor -
    MyUCB1AlgorithmModel() = new();
end

mutable struct MyBinaryBanditGreedyAlgorithmModel <: AbstractBanditAlgorithmModel

    # data -
    K::Int64 # number of arms (each arm evalutes to a binary reward)
    S::Dict{Int64, Int64} # success
    F::Dict{Int64, Int64} # failure

    # constructor -
    MyBinaryBanditGreedyAlgorithmModel() = new();
end

mutable struct MyContextualBernoulliBanditAlgorithmModel <: AbstractBanditAlgorithmModel

    # data -
    K::Dict{Int64, Int64} # number of arms
    bandits::Dict{Int64, MyBinaryBanditGreedyAlgorithmModel} # bandits
    keys::Vector{Int64} # keys

    # constructor -
    MyContextualBernoulliBanditAlgorithmModel() = new();
end

mutable struct MyContextualBernoulliBanditAlgorithmWeatherContextModel <: AbstractBanditAlgorithmContextModel

    # data -
    d::Normal # distribution for high temperatures
    key::Int64 # context key
    data::DataFrame

    # constructor -
    MyContextualBernoulliBanditAlgorithmWeatherContextModel() = new();
end

struct MyEmptyContextModel <: AbstractBanditAlgorithmContextModel
    MyEmptyContextModel() = new(); # constructor
end