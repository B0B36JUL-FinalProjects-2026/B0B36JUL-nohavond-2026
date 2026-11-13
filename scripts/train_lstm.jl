include("common.jl")
include("plots.jl")

Random.seed!(SEED)

train, val, test = prepare_data()

lstm_point = LSTMModel(hidden = 64, dense = 32, epochs = 30)
lstm_gauss = LSTMModel(head = GaussianHead(), hidden = 64, dense = 32, epochs = 30)

models = [
    "LSTM" => lstm_point,
    "LSTM + nejistota" => lstm_gauss,
]

for (name, model) in models
    println("Trénuji: $name")
    fit!(model, train; val)
end

save_results(models, val, test, "lstm")
save_models("lstm"; point_model = lstm_point, gauss_model = lstm_gauss)
network_figure(lstm_gauss, test, "lstm.png")
