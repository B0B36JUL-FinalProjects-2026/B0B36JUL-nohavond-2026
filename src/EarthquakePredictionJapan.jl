module EarthquakePredictionJapan

using CSV
using DataFrames
using Dates
using Statistics

export GeoBounds, JAPAN, load_catalog, clean_catalog
export DataSplit, chronological_split
export WINDOW_FEATURES, WindowedData, make_windows, flatten_windows
export Standardizer, fit_standardizer

include("data/load.jl")
include("data/split.jl")
include("data/windows.jl")
include("data/scaling.jl")

end