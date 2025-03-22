
function simulate(model::MySimpleBoltzmannMachineModel, sₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)::Tuple{Vector{Int}, Array{Int,2}, Array{Float64,1}}
    
    # initialize storage -
    W = model.W; # weight matrix
    b = model.b; # bias vector

    number_of_neurons = length(sₒ);
    S = zeros(Int, number_of_neurons, T);
    turns = zeros(Int, T);
    energy = zeros(Float64, T);
    is_ok_to_stop = false; # flag to stop the simulation

    # package initial state -
    s = copy(sₒ); # initial state
    energy[1] = -(1/2)*dot(s, W*s) - dot(b, s); # compute the energy of the initial state
    S[:, 1] .= s; # store the initial state in the S matrix
    turns[1] = 1; # store the time step

    # main loop -
    t = 2;
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

function decode(simulationstate::Array{T,1}; 
    number_of_rows::Int64 = 28, number_of_cols::Int64 = 28)::Array{T,2} where T <: Number
    
    # initialize -
    reconstructed_image = Array{Int32,2}(undef, number_of_rows, number_of_cols);
    linearindex = 1;
    for row ∈ 1:number_of_rows
        for col ∈ 1:number_of_cols
            s = simulationstate[linearindex];
            if (s == -1)
                reconstructed_image[row,col] = 0;
            else
                reconstructed_image[row,col] = 1;
            end
            linearindex+=1;
        end
    end
    
    # return 
    return reconstructed_image
end