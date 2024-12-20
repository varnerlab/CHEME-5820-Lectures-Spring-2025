

function _cluster(data::Array{Float64,2}, algorithm::MyNaiveKMeansClusteringAlgorithm)
    
    # get data -
    K = algorithm.K;
    ϵ = algorithm.ϵ;
    maxiter = algorithm.maxiter;
    assigments = algorithm.assigments;
    centriods = algorithm.centriods;
    dimension = algorithm.dimension;
    number_of_points = algorithm.number_of_points;
    loopcount = 1; # how many iterations have we done?\
    tmp = zeros(Float64, K);
    
    # main -
    has_converged = false; # convergence flag
    while (has_converged == false)
    
        # step 1: assign each data point to the nearest centriod -
        for i ∈ 1:number_of_points
            for k ∈ 1:K
                tmp[k] = euclidean(data[i,:], centriods[k]);
            end
            assigments[i] = argmin(tmp);
        end
    
        # step 2: update the centroids -
        for k ∈ 1:K
            centriods[k] = mean(data[assigments .== k, :], dims=1);
        end

        # check: have we reached the maximum number of iterations -or- have the centroids converged?
        if (loopcount > maxiter)
            has_converged = true;
        else
            loopcount += 1; # update the loop count
        end
    end
    
    # return the model -
    return algorithm;
end

function cluster(data::Array{Float64,2}, algorithm::T) where T <: MyAbstractUnsupervisedClusteringAlgorithm
    return _cluster(data, algorithm);
end