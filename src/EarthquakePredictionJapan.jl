module EarthquakePredictionJapan

using CSV
using DataFrames
using Dates
using Random
using Statistics
using Distributions: Normal, cdf, pdf, logpdf
import DecisionTree
import Flux

export GeoBounds, JAPAN, load_catalog, clean_catalog
export DataSplit, chronological_split
export WINDOW_FEATURES, WindowedData, make_windows, flatten_windows
export Standardizer, fit_standardizer

export AbstractPrediction, ProbabilisticPrediction
export PointPrediction, SamplePrediction, GaussianPrediction, point, interval
export AbstractMetric, MAE, RMSE, CRPS, NLL, Coverage, evaluate

export AbstractQuakeModel, AbstractHead, PointHead, SampleHead, GaussianHead, fit!, predict
export MeanBaseline, RandomForest, MLP

include("data/load.jl")
include("data/split.jl")
include("data/windows.jl")
include("data/scaling.jl")

include("predictions.jl")
include("metrics.jl")

include("models/interface.jl")
include("losses.jl")
include("train_flux.jl")

include("models/baseline.jl")
include("models/forest.jl")
include("models/mlp.jl")

end