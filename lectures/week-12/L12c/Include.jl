# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_DATA = joinpath(_ROOT, "data");

# load external packages -
using Pkg
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# using statements -
using Plots
using Colors
using LinearAlgebra
using Statistics
using NNlib
using Distributions
using DataFrames
using FileIO
using CSV
using DataFramesMeta
using Optim
using Flux

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
# include(joinpath(_PATH_TO_SRC, "Factory.jl"));
# include(joinpath(_PATH_TO_SRC, "Compute.jl"));
# include(joinpath(_PATH_TO_SRC, "Train.jl"));