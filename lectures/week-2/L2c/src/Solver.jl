
"""
    solve(problem::MyLinearProgrammingProblemModel) -> Dict{String,Any}

Solves a linear programming problem. This function uses the GLPK optimizer to solve the problem, 
and returns the optimal choice and the value of the objective function at the optimal choice.
This function throws an AssertionError if the optimization is not successful.

### Arguments
- problem::MyLinearProgrammingProblemModel: An instance of MyLinearProgrammingProblemModel holding the data for the problem.

### Returns
- Dict{String,Any}: A dictionary with the following keys:
    - "argmax": The optimal choice.
    - "objective_value": The value of the objective function at the optimal choice.
"""
function solve(problem::MyLinearProgrammingProblemModel)::Dict{String,Any}

    # initialize -
    results = Dict{String,Any}()
    c = problem.c; # objective function coefficients
    lb = problem.lb; # lower bounds
    ub = problem.ub; # upper bounds
    A = problem.A; # constraint matrix
    b = problem.b; # right-hand side vector

    # how many variables do we have?
    d = length(c);

    # Setup the problem -
    model = Model(GLPK.Optimizer)
    @variable(model, lb[i,1] <= x[i=1:d] <= ub[i,1], start=0.0) # we have d variables
    
    # inq constraints: A*x <= b
    index_ineq = [11,12,13,14,15]; # source and sink nodes
    Aineq = A[index_ineq,:];
    bineq = b[index_ineq];

    # eq constraints: A*x = b
    index_eq = [1,2,3,4,5,6,7,8,9,10]; # machine and product nodes
    Aeq = A[index_eq,:];
    beq = b[index_eq];

    # set objective function -   
    @objective(model, Max, transpose(c)*x);
    @constraints(model,
        begin
            Aineq*x .<= bineq # my inequality constraints
            Aeq*x .== beq # my equality constraints
        end
    );

    # run the optimization -
    optimize!(model)

    # check: was the optimization successful?
    @assert is_solved_and_feasible(model) == true

    # populate -
    x_opt = value.(x);
    results["argmax"] = x_opt
    results["objective_value"] = objective_value(model);

    # return -
    return results
end