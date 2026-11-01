module EarthquakePredictionJapan

using CSV
using DataFrames
using Dates
using Statistics

export GeoBounds, JAPAN, load_catalog, clean_catalog
export DataSplit, chronological_split

include("data/load.jl")
include("data/split.jl")
# include("data/windows.jl")
# include("data/scaling.jl")

end