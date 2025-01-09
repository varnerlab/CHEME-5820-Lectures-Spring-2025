function build(modeltype::Type{MyPerceptronClassificationModel}, data::NamedTuple)

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