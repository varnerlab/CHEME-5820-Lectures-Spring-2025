"""
    build(modeltype::Type{MyPerceptronClassificationModel}, 
        data::NamedTuple) -> MyPerceptronClassificationModel

The function builds a perceptron classification model from the data provided.

### Arguments
- `modeltype::Type{MyPerceptronClassificationModel}`: the type of the model to build.
- `data::NamedTuple`: the data to use to build the model.

The `data::NamedTuple` must have the following fields:
- `parameters::Vector{Float64}`: the coefficients of the model.
- `mistakes::Int64`: the number of mistakes that are are willing to make.

### Returns
- a perceptron classification model.
"""
function build(modeltype::Type{MyPerceptronClassificationModel}, 
    data::NamedTuple)::MyPerceptronClassificationModel

    # build an empty model -
    model = modeltype();
    β = data.parameters;
    m = data.mistakes;
    
    # set the data -
    model.β = β;
    model.mistakes = m;

    # return -
    return model;
end

"""
    build(modeltype::Type{MyLogisticRegressionClassificationModel}, 
        data::NamedTuple) -> MyLogisticRegressionClassificationModel

The function builds a logistic regression classification model from the data provided.

### Arguments
- `modeltype::Type{MyLogisticRegressionClassificationModel}`: the type of the model to build.
- `data::NamedTuple`: the data to use to build the model.

The `data::NamedTuple` must have the following fields:
- `parameters::Vector{Float64}`: the coefficients of the model.

### Returns
- a logistic regression classification model.
"""
function build(modeltype::Type{MyLogisticRegressionClassificationModel}, 
    data::NamedTuple)::MyLogisticRegressionClassificationModel

    # build an empty model -
    model = modeltype();
    β = data.parameters;
    α = data.learning_rate
    L = data.loss_function
    ϵ = data.ϵ
    
    # set the data -
    model.β = β;
    model.α = α;
    model.L = L;
    model.ϵ = ϵ;

    # return -
    return model;
end

"""
    build(modeltype::Type{MyKNNClassificationModel}, 
        data::NamedTuple) -> MyKNNClassificationModel

The function builds a KNN classification model from the data provided.

### Arguments
- `modeltype::Type{MyKNNClassificationModel}`: the type of the model to build.
- `data::NamedTuple`: the data to use to build the model.

The `data::NamedTuple` must have the following fields:
- `K::Int64`: the number of neighbours to look at.
- `d::Function`: the similarity function.
- `X::Matrix{Float64}`: the training features.
- `y::Vector{Int64}`: the training labels.
"""
function build(modeltype::Type{MyKNNClassificationModel}, 
    data::NamedTuple)::MyKNNClassificationModel

    # build an empty model -
    model = modeltype();
    K = data.K;
    d = data.d;
    features = data.features;
    labels = data.labels;
    
    # set the data -
    model.K = K;
    model.d = d;
    model.X = features;
    model.y = labels;

    # return -
    return model;
end