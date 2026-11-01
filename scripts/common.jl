using EarthquakePredictionJapan
using Printf
using Random

const SEED = 42
const K = 10
const MODELS_DIR = normpath(joinpath(@__DIR__, "..", "models"))
const FIGURES_DIR = normpath(joinpath(@__DIR__, "..", "figures"))
const METRICS = [MAE(), RMSE(), CRPS(), Coverage(0.9)]

function prepare_data(; k = K)
    df = load_catalog()
    parts = chronological_split(df)
    train, val, test = make_windows.((parts.train, parts.val, parts.test); k)
    println("Okna: train $(length(train)), val $(length(val)), test $(length(test))")
    return train, val, test
end

fmt(x::Real) = @sprintf("%.3f", x)
fmt(::Missing) = "–"

function results_table(models, data, title; metrics = METRICS)
    println("\n", title)
    println(rpad("Model", 28), join(rpad.(string.(metrics), 14)))
    for (name, model) in models
        p = predict(model, data)
        println(rpad(name, 28), join(rpad.([fmt(evaluate(m, p, data.y)) for m in metrics], 14)))
    end
end