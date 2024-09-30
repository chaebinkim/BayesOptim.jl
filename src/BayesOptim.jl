module BayesOptim
using Pkg
# Pkg.build("PyCall") 
using PyCall
using CSV, CairoMakie, DataFrames

include("SafetyChecks.jl")
include("Fit.jl")

export Fit

end
