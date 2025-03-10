function build(modeltype::Type{MyEpsilonGreedyAlgorithmModel}, data::NamedTuple)

    # initialize -
    K = data.K; # number of arms

    # build empty model -
    model = modeltype();
    model.K = K;

    # return -
    return model;
end

function build(modeltype::Type{MyExploreFirstAlgorithmModel}, data::NamedTuple)

    # initialize -
    K = data.K; # number of arms

    # build empty model -
    model = modeltype();
    model.K = K;

    # return -
    return model;
end

function build(modeltype::Type{MyUCB1AlgorithmModel}, data::NamedTuple)

    # initialize -
    K = data.K; # number of arms

    # build empty model -
    model = modeltype();
    model.K = K;

    # return -
    return model;
end

function build(modeltype::Type{MyBinaryBanditGreedyAlgorithmModel}, data::NamedTuple)

    # initialize -
    K = data.K; # number of arms

    # build empty model -
    model = modeltype();
    model.K = K;
    model.S = Dict{Int64, Int64}(); # initialize success
    model.F = Dict{Int64, Int64}(); # initialize failure

    # initialize success and failure dictionaries for each arm
    for i ∈ 1:K
        model.S[i] = 0; # initialize success
        model.F[i] = 0; # initialize failure
    end

    # return -
    return model;
end

function build(modeltype::Type{MyContextualBernoulliBanditAlgorithmModel}, data::NamedTuple)

    # initialize -
    K = data.K; # arms dictionary
    keys = data.keys; # keys (external order of the arms) # ints
    bandits = Dict{Int64, MyBinaryBanditGreedyAlgorithmModel}(); # empty bandits dictionary

    # build empty model -
    model = modeltype();

    # initialize bandits for each arm
    for key ∈ keys
        
        # build an empty bandit -
        bandit = build(MyBinaryBanditGreedyAlgorithmModel, (
            K = K[key], # number of arms
        ));
        
        # get the data for this bandit -
        bandits[key] = bandit; # add to bandits dictionary
    end

    # store the bandits -
    model.K = K; # store number the arms dictionary
    model.bandits = bandits; # store the bandits
    model.keys = keys; # store the keys

    # return -
    return model;
end

function build(modeltype::Type{MyContextualBernoulliBanditAlgorithmWeatherContextModel}, 
    data::NamedTuple)

    # initialize -
    d = data.d; # distribution for high temperatures
    key = data.key; # context key
    dataframe = data.dataframe; # data

    # build empty model -
    model = modeltype();
    model.d = d;
    model.key = key;
    model.data = dataframe;

    # return -
    return model;
end