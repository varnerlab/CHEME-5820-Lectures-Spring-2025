abstract type AbstractTextRecordModel end
abstract type AbstractTextDocumentCorpusModel end
abstract type AbstractNeuralNetwork end

# build a simple neural network model type -
struct MyFluxNeuralNetworkModel <: AbstractNeuralNetwork
    chain::Chain; # holds the model chain
end

"""
    MySarcasmRecordModel <: AbstractTextRecordModel

### Fields 
- `data::Array{String, Any}`: The data found in the record in the order they were found
"""
mutable struct MySarcasmRecordModel <: AbstractTextRecordModel
    
    # data -
    issarcastic::Bool
    headline::String
    article::String
    
    # constructor -
    MySarcasmRecordModel() = new(); # empty
end

"""
    MySarcasmRecordCorpusModel <: AbstractTextDocumentCorpusModel

### Fields
- `records::Dict{Int, MySarcasmRecordModel}`: The records in the document (collection of records)
- `tokens::Dict{String, Int64}`: A dictionary of tokens in alphabetical order (key: token, value: position) for the entire document
"""
mutable struct MySarcasmRecordCorpusModel <: AbstractTextDocumentCorpusModel
    
    # data -
    records::Dict{Int, MySarcasmRecordModel}
    tokens::Dict{String, Int64}
    inverse::Dict{Int64, String}
    
    # constructor -
    MySarcasmRecordCorpusModel() = new(); # empty
end

mutable struct MyCBOWEmbeddingModel

    # data -
    W₁::Array{<:Number, 2} # input layer weights 
    W₂::Array{<:Number, 2} # output layer weights
    din::Int64 # input layer dimension
    h::Int64 # hidden layer dimension
    dout::Int64 # output layer dimension

    # empty constructor -
    MyCBOWEmbeddingModel() = new(); # empty
end