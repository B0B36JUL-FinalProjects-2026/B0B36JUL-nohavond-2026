module EarthquakePredictionJapan

using CSV
using DataFrames
using Dates
using Statistics
import DecisionTree

export GeoBounds, JAPAN, load_catalog, clean_catalog
export DataSplit, chronological_split
export WINDOW_FEATURES, WindowedData, make_windows, flatten_windows
export Standardizer, fit_standardizer

export AbstractPrediction, PointPrediction, SamplePrediction, point, interval
export AbstractMetric, MAE, RMSE, CRPS, Coverage, evaluate

export AbstractQuakeModel, AbstractHead, PointHead, SampleHead, fit!, predict
export MeanBaseline

include("data/load.jl")
include("data/split.jl")
include("data/windows.jl")
include("data/scaling.jl")

include("predictions.jl")
include("metrics.jl")

include("models/interface.jl")
include("models/baseline.jl")
# include("models/forest/forest.jl")

end