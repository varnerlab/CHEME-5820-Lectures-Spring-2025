function build(modeltype::Type{MyNaiveKMeansClusteringAlgorithm}, data::NamedTuple)::MyNaiveKMeansClusteringAlgorithm
    
    # build an empty model -
    model = modeltype();

    # get data -
    K = data.K;
    ϵ = data.ϵ;
    maxiter = data.maxiter;
    dimension = data.dimension;
    number_of_points = data.number_of_points;

    # setup the initial assigments -
    assigments = zeros(Int64, number_of_points);
    for i ∈ 1:number_of_points
        assigments[i] = rand(1:K); # randomly assign points to clusters
    end

    # setup the centriods -
    centriods = Dict{Int64, Vector{Float64}}();
    for k ∈ 1:K
        centriods[k] = rand(Float64, dimension); # randomly generate the centriods
    end

    # set the data on the model -
    model.K = K;
    model.ϵ = ϵ;
    model.maxiter = maxiter;
    model.dimension = dimension;
    model.number_of_points = number_of_points;
    model.assigments = assigments;
    model.centriods = centriods;

    # return the model -
    return model;
end