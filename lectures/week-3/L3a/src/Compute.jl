
function _learn(features::Array{<:Number,2}, labels::Array{<:Number,1}, algorithm::MyPerceptronClassificationModel)
end

function learn(features::Array{<:Number,2}, labels::Array{<:Number,1, algorithm::T}) where T <: AbstractClassificationAlgorithm
    return _learn(features, labels, algorithm);
end
   