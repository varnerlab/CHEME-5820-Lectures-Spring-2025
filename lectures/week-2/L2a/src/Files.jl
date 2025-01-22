function _jld2(path::String)::Dict{String,Any}
    return load(path);
end

# -- PUBLIC METHODS BELOW HERE ------------------------------------------------------------------------------------------------- #
"""
    MySyntheticDataset(;visit::Int = 1)

Load the synthetic coagulation dataset for the specified visit.

### Arguments
- `visit::Int = 1`: The visit number to load. Visits are numbered from 1, 2 or 3 and correspond to the three visits
  of the synthetic patient: 1 = baseline (non-pregnant), 2 = first trimester, 3 = third trimester.

### Returns
- A dictionary containing the original patient index as keys, the corresponding synthetic patient data as values.
"""
MySyntheticDataset(;visit::Int = 1) = _jld2(joinpath(_PATH_TO_DATA, "SyntheticPatient-MeasurementEnsemble-S90-V$(visit).jld2"));
# --- PUBLIC METHODS ABOVE HERE ------------------------------------------------------------------------------------------------ #