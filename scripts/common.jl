using EarthquakePredictionJapan
using Printf
using Random

const SEED = 42
const K = 10
const INTERVAL = 0.9
const MODELS_DIR = normpath(joinpath(@__DIR__, "..", "models"))
const FIGURES_DIR = normpath(joinpath(@__DIR__, "..", "figures"))
const RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results"))
const METRICS = [MAE(), RMSE(), CRPS(), Coverage(INTERVAL)]

function prepare_data(; k = K)
    df = load_catalog()
    parts = chronological_split(df)
    train, val, test = make_windows.((parts.train, parts.val, parts.test); k)
    return train, val, test
end

fmt(x::Real) = @sprintf("%.3f", x)
fmt(::Missing) = "–"

function results_table(models, data, title; metrics = METRICS, io = stdout)
    println(io, "\n", title)
    println(io, rpad("Model", 28), join(rpad.(string.(metrics), 14)))
    for (name, model) in models
        p = predict(model, data)
        println(io, rpad(name, 28), join(rpad.([fmt(evaluate(m, p, data.y)) for m in metrics], 14)))
    end
end

function save_results(models, val, test, name)
    results_table(models, val, "Validační data")
    results_table(models, test, "Testovací data")

    mkpath(RESULTS_DIR)
    file = joinpath(RESULTS_DIR, "$name.txt")
    open(file, "w") do io
        results_table(models, val, "Validační data"; io)
        results_table(models, test, "Testovací data"; io)
    end
end