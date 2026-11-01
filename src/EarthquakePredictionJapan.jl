module EarthquakePredictionJapan 

using CSV
using DataFrames
using Dates
 
# Data
export GeoBounds, JAPAN, load_catalog, clean_catalog
include("data/load.jl")



end