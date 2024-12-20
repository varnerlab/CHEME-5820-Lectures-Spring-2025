

function _cluster(data::Array{<:Number,2}, algorithm::MyNaiveKMeansClusteringAlgorithm; 
    d = Euclidean(), verbose::Bool = false)
    
    # get data -
    K = algorithm.K;
    ϵ = algorithm.ϵ;
    maxiter = algorithm.maxiter;
    assignments = algorithm.assignments;
    centroids = algorithm.centroids;
    dimension = algorithm.dimension;
    number_of_points = algorithm.number_of_points;
    loopcount = 1; # how many iterations have we done?\
    tmp = zeros(Float64, K);
    
    # main -
    has_converged = false; # convergence flag
    while (has_converged == false)
    
        # step 1: assign each data point to the nearest centriod -
        â = copy(assignments); # old assignments
        for i ∈ 1:number_of_points
            for k ∈ 1:K
                tmp[k] = d(data[i,:], centroids[k]);
            end
            assignments[i] = argmin(tmp);
        end
    
        # step 2: update the centroids -
        for k ∈ 1:K
            index_cluter_k = findall(x-> x == k, assignments); # index of the data vectors assigned to cluster k

            if (isempty(index_cluter_k) == true)
                continue;
            else
                for d ∈ 1:dimension
                    centroids[k][d] = mean(data[index_cluter_k, d]);
                end
            end
        end

        # verbose -
        if (verbose == true)
            # if we get here, we are in verbose mode. Dump results to disk
            # ...
        end

        # check: have we reached the maximum number of iterations -or- have the centroids converged?
        if (loopcount > maxiter || d(â, assignments) ≤ ϵ)
            has_converged = true;
        else
            loopcount += 1; # update the loop count
        end
    end
    
    # return the model -
    return (algorithm.assignments, algorithm.centroids, loopcount);
end

function cluster(data::Array{<:Number,2}, algorithm::T; d = Euclidean(), verbose::Bool = false) where T <: MyAbstractUnsupervisedClusteringAlgorithm
    return _cluster(data, algorithm, d = d, verbose = verbose);
end