module BayesOptim
using Pkg
# Pkg.build("PyCall") 
using PyCall
import CSV 
import CairoMakie
import DataFrames

include("SafetyChecks.jl")
include("Fit.jl")

export Fit

end
