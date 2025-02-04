# --- PRIVATE API BELOW HERE -------------------------------------------------------------------------------------- #
function _learn(features::Array{<:Number,2}, labels::Array{<:Number,1}, algorithm::MyPerceptronClassificationModel; 
    maxiter::Int64 = 100, verbose::Bool = false)

    # get data from the algorithm -
    β = algorithm.β;
    m = algorithm.mistakes;
    is_ok_to_continue = true;
    loop_counter = 1;
    error_counter = 0;
    
    # main loop -
    β̂ = copy(β); # copy the coefficients
    while (is_ok_to_continue == true) 
        
        error_counter = 0; # initialize the error counter
        for i ∈ eachindex(labels) # for each training pair
            
            x = features[i,:]; # feature vector (n+1) x 1
            y = labels[i]; # classification -1,1
            
            # check: misclassified?
            if (y*sum(β̂.*x)) ≤ 0
                β̂ = β̂ .+ y*x;
                error_counter+=1        
            end
        end # end training loop for

        # should we stay in the loop? (or should we go ...)
        if (loop_counter >= maxiter || error_counter ≤ m)
            is_ok_to_continue = false; # we are done!
        else
            is_ok_to_continue = true; # we are not done yet!
            loop_counter+=1; # increment the loop counter
        end
    end

    # print the results if verbose is true -
    if (verbose == true)
        println("Stopped after number of iterations: ", loop_counter, ". We have number of errors: ", error_counter);
    end
    
    # update the model -
    algorithm.β = β̂; # update the coefficients

    # return -
    return algorithm;
end

function _learn(features::Array{<:Number,2}, labels::Array{<:Number,1}, algorithm::MyLogisticRegressionClassificationModel; 
    maxiter::Int64 = 100, verbose::Bool = false)
    
    # get data from the algorithm -
    β = algorithm.β; # parameters
    α = algorithm.α; # learning rate
    ϵ = algorithm.ϵ; # stopping criterion
    L = algorithm.L; # loss function
    is_ok_to_continue = true;
    loop_counter = 1;

    # main loop -
    βᵢ = copy(β); # copy the coefficients, initial guess
    error = nothing;
    while (is_ok_to_continue == true && loop_counter ≤ maxiter) 
        
        # initialize the gradient -
        ∇L = zeros(size(βᵢ));
        
        # compute the gradient -
        for i ∈ eachindex(labels) # for each training pair
            
            x = features[i,:]; # feature vector (n+1) x 1
            y = labels[i]; # classification -1,1
            
            # compute the gradient -
            ∇L = ∇L .+ ∇f(L, x, y, βᵢ, 1e-6); # call forward diff approx of derivative
        end # end training loop for
        
        # update the coefficients -
        βᵢ = βᵢ .- α*∇L;
        
        # should we stay in the loop? (or should we go ...)
        error = norm(∇L);
        if (error ≤ ϵ)
            is_ok_to_continue = false; # we are done!
        else
            is_ok_to_continue = true; # we are not done yet!
            loop_counter+=1; # increment the loop counter
        end
    end

    # print the results if verbose is true -
    if (verbose == true)
        println("Stopped after number of iterations: ", loop_counter, ". We have error: ", error);
    end

    # update the algorithm -
    algorithm.β = βᵢ; # update the coefficients

    # return -
    return algorithm;
end

function _classify(features::Array{<:Number,2}, algorithm::MyPerceptronClassificationModel)
    return sign.(features*algorithm.β);
end
# --- PRIVATE API ABOVE HERE -------------------------------------------------------------------------------------- #

# --- PUBLIC API BELOW HERE --------------------------------------------------------------------------------------- #
"""
    learn(features::Array{<:Number,2}, labels::Array{<:Number,1}, algorithm::AbstractClassificationAlgorithm; 
        maxiter::Int64 = 100, verbose::Bool = false)
"""
function learn(features::Array{<:Number,2}, labels::Array{<:Number,1}, algorithm::AbstractClassificationAlgorithm; 
    maxiter::Int64 = 100, verbose::Bool = false)
    
    # call the internal function, and return the updated algorithm model
    return _learn(features, labels, algorithm, maxiter = maxiter, verbose = verbose);
end

"""
    classify(features::Array{<:Number,2}, algorithm::AbstractClassificationAlgorithm)
"""
function classify(features::Array{<:Number,2}, algorithm::AbstractClassificationAlgorithm)
    return _classify(features, algorithm);
end

# --- PUBLIC API ABOVE HERE --------------------------------------------------------------------------------------- #