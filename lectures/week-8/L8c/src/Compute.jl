# PRIVATE METHODS BELOW HERE ================================================================================= #
# placeholder - always return 0
_null(action::Int64)::Int64 = return 0;


function _world(model::MyRectangularGridWorldModel, s::Int, a::Int)

    # initialize -
    s′ = nothing
    r = nothing
    
    # get data from the model -
    coordinates = model.coordinates;
    moves = model.moves
    states = model.states;
    rewards = model.rewards;

    # where are we now?
    current_position = coordinates[s];

    # get the perturbation -
    Δ = moves[a];
    new_position = current_position .+ Δ

    # before we go on, have we "driven off the grid"?
    if (haskey(states, new_position) == true)

        # lookup the new state -
        s′ = states[new_position];
        r = rewards[s′];
    else
       
        # ok: so we are all the grid. Bounce us back to to the current_position, and charge a huge penalty 
        s′ = states[current_position];
        r = -1000000000000.0
    end

    # return -
    return (s′,r);
end
# PRIVATE METHODS ABOVE HERE ================================================================================= #

# PUBLIC METHODS BELOW HERE ================================================================================== #``

"""
    Fill me in
"""
function solve(agent::MyQLearningAgentModel, environment::MyRectangularGridWorldModel; 
    maxsteps::Int = 100, δ::Float64 = 0.02)::MyQLearningAgentModel

    # initialize -
    actions = agent.actions;
    K = length(actions); # number of actions
    states = agent.states;
    Q₁ = agent.Q;
    γ = agent.γ;

    # simulation loop -
    for s ∈ states

        # initialize t -
        t = 1;
        has_converged = false;
        αₜ = copy(agent.α);   
        while (has_converged == false)
            
            # compute the ϵ -
            ϵₜ = (1.0/(t^(1/3)))*(log(K*t))^(1/3); # compute the epsilon value -
            p = rand();

            aₜ = nothing;
            if p ≤ ϵₜ
                aₜ = rand(1:K); # generate a random action
            elseif p > ϵₜ
                aₜ = argmax(Q₁[s,:]); # select the greedy action, given state s
            end

            # compute new state and reward -
            s′, r = _world(environment, s, aₜ);
            
            # use the update rule to update Q -
            Q₂ = copy(Q₁); # this seems really inefficient, but it is what it is ...
            Q₁[s,aₜ] += αₜ*(r+γ*maximum(Q₁[s′,:]) - Q₁[s,aₜ])

            # update stuff
            s = s′; # state update
            t += 1; # time update
            αₜ = 0.99*αₜ; # update the learning rate

            # check if we have converged -
            if ((t > maxsteps) || norm(Q₂ - Q₁) < δ)
                has_converged = true;
            end
        end
    end

    agent.Q = Q₁; # update the model

    # return -
    return agent
end

function policy(Q_array::Array{Float64,2})::Array{Int64,1}

    # get the dimension -
    (NR, _) = size(Q_array);

    # initialize some storage -
    π_array = Array{Int64,1}(undef, NR)
    for s ∈ 1:NR
        π_array[s] = argmax(Q_array[s,:]);
    end

    # return -
    return π_array;
end
# PUBLIC METHODS ABOVE HERE ================================================================================== #