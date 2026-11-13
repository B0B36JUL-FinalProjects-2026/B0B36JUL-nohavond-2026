using EarthquakePredictionJapan
using JLD2
using Printf
using Random

const SEED = 42
const K = 10
const INTERVAL = 0.9
const MODELS_DIR = normpath(joinpath(@__DIR__, "..", "models"))
const FIGURES_DIR = normpath(joinpath(@__DIR__, "..", "figures"))
const RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results"))
const METRICS = [MAE(), RMSE(), CRPS(), NLL(), Coverage(INTERVAL)]

function prepare_data(; k = K)
    windows = make_windows(load_catalog(); k)
    parts = chronological_split(windows)
    train, val, test = parts.train, parts.val, parts.test
    println("Okna: train $(length(train)), val $(length(val)), test $(length(test))")
    return train, val, test
end

fmt(x::Real) = @sprintf("%.3f", x)
fmt(::Missing) = "–"

function results_table(predictions, y, title; metrics = METRICS, io = stdout)
    println(io, "\n", title)
    println(io, rpad("Model", 28), join(rpad.(string.(metrics), 14)))
    for (name, p) in predictions
        println(io, rpad(name, 28), join(rpad.([fmt(evaluate(m, p, y)) for m in metrics], 14)))
    end
end

function save_results(models, val, test, run_name)
    val_predictions = [name => predict(model, val) for (name, model) in models]
    test_predictions = [name => predict(model, test) for (name, model) in models]

    report(io) = begin
        results_table(val_predictions, val.y, "Validační data"; io)
        results_table(test_predictions, test.y, "Testovací data"; io)
    end
    report(stdout)

    mkpath(RESULTS_DIR)
    open(report, joinpath(RESULTS_DIR, "$run_name.txt"), "w")
    jldsave(joinpath(RESULTS_DIR, "$run_name.jld2");
        names = first.(models),
        val_predictions = Dict(val_predictions), test_predictions = Dict(test_predictions),
        val_y = val.y, test_y = test.y)
end

# Neuronové sítě se ukládají jen jako Flux.state a parametry škálování
# (viz model_state), ostatní modely celé.
saved_form(m::NetworkModel) = model_state(m)
saved_form(m::AbstractQuakeModel) = m

function save_models(run_name; models...)
    mkpath(MODELS_DIR)
    file = joinpath(MODELS_DIR, "$run_name.jld2")
    jldsave(file; k = K, (name => saved_form(m) for (name, m) in models)...)
end
