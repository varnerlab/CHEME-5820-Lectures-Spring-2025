
function simulate(model::MySimpleBoltzmannMachineModel, sₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)::Tuple{Vector{Int}, Array{Int,2}, Array{Float64,1}}
    
    # initialize storage -
    W = model.W; # weight matrix
    b = model.b; # bias vector

    number_of_neurons = length(sₒ);
    S = zeros(Int, number_of_neurons, T+1);
    turns = zeros(Int, T);
    energy = zeros(Float64, T);
    is_ok_to_stop = false; # flag to stop the simulation
    t = 1; # time step

    # main loop -
    t = 1;
    s = copy(sₒ);
    while (is_ok_to_stop == false)

        # process each neuron -
        for i ∈ 1:number_of_neurons
            hᵢ = dot(W[i, :], s) + b[i]; # compute the input for node i -
            pᵢ = 1 / (1 + exp(-2 * β * hᵢ));  # compute the probability of flipping the i-th bit
            flag = Bernoulli(pᵢ) |> rand; # build a bernoulli random variable, sample it
            s[i] = flag == 1 ? 1 : -1; # flip the i-th bit
        end
        
        energy[t] = -(1/2)*dot(s, W*s) - dot(b, s); # compute and store the energy of the current state
        S[:, t] .= copy(s); # store the current state in the S matrix
        turns[t] = t; # store the time step
        
        if (t == T)
            is_ok_to_stop = true; # stop the simulation
        else
            t += 1; # increment the time step, and continue
        end
    end

    # return the results -    
    return (turns, S, energy);
end