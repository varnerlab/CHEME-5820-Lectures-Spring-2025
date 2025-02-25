function _jld2(path::String)::Dict{String,Any}
    return load(path);
end

MyMarketDataSet() = _jld2(joinpath(_PATH_TO_DATA, "SP500-Daily-OHLC-1-3-2014-to-02-07-2025.jld2"));