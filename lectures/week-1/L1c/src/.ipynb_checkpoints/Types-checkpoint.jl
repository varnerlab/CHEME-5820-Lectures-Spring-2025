abstract type MyAbstractUnsupervisedClusteringAlgorithm end


mutable struct MyNaiveKMeansClusteringAlgorithm <: MyAbstractUnsupervisedClusteringAlgorithm

    # data -
    K::Int64 # number of clusters
    centroids::Dict{Int64, Vector{Float64}} # cluster centroids
    assignments::Vector{Int64} # cluster assignments
    ϵ::Float64 # convergence criteria
    maxiter::Int64 # maximum number of iterations (alternatively, could use this convergence criterion)
    dimension::Int64 # dimension of the data
    number_of_points::Int64 # number of data points

    # constructor -
    MyNaiveKMeansClusteringAlgorithm() = new(); # build empty object
end