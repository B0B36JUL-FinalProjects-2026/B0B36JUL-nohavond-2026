include("common.jl")
using CairoMakie

Random.seed!(SEED)

train, val, test = prepare_data()

lstm_point = LSTMModel(hidden = 64, dense = 32, epochs = 50)
lstm_gauss = LSTMModel(head = GaussianHead(), hidden = 64, dense = 32, epochs = 50)

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

gauss_pred = predict(lstm_gauss, test)
lower, upper = interval(gauss_pred, INTERVAL)
pct = round(Int, 100 * INTERVAL)

fig = Figure(size = (1100, 450))

ax1 = Axis(fig[1, 1]; title = "Průběh tréninku", xlabel = "Epocha", ylabel = "Loss")
lines!(ax1, lstm_gauss.history; label = "train (NLL)")
lines!(ax1, lstm_gauss.val_history; label = "val (NLL)")
axislegend(ax1)

ax2 = Axis(fig[1, 2]; title = "Predikce s $(pct)% intervalem (posledních 200 událostí)",
    xlabel = "Pořadí události v testu", ylabel = "Magnitudo")
idx = max(1, length(test) - 199):length(test)
band!(ax2, idx, lower[idx], upper[idx]; color = (:darkorange, 0.25), label = "$(pct)% interval")
lines!(ax2, idx, point(gauss_pred)[idx]; color = :darkorange, label = "predikce")
scatter!(ax2, idx, test.y[idx]; color = :black, markersize = 5, label = "skutečnost")
axislegend(ax2; position = :lt)

mkpath(FIGURES_DIR)
figure_file = joinpath(FIGURES_DIR, "lstm.png")
save(figure_file, fig)