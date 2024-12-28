function _jld2(path::String)::Dict{String,Any}
    return load(path);
end

# -- PUBLIC METHODS BELOW HERE ------------------------------------------------------------------------------------------------- #
MySyntheticDataset(;visit::Int = 1) = _jld2(joinpath(_PATH_TO_DATA, "SyntheticPatient-MeasurementEnsemble-S90-V$(visit).jld2"));
# --- PUBLIC METHODS ABOVE HERE ------------------------------------------------------------------------------------------------ #