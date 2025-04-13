abstract type AbstractLayerModel end

struct MyFluxElmanRecurrentNeuralNetworkModel <: AbstractLayerModel
    chain::Flux.Chain; # holds the model chain
end

struct MyFluxLSTMNeuralNetworkModel <: AbstractLayerModel
    chain::Flux.Chain; # holds the model chain
end