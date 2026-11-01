include("common.jl")
using CairoMakie

Random.seed!(SEED)

train, val, test = prepare_data()

mlp_point = MLP(hidden = [64, 32, 16], epochs = 20)
mlp_gauss = MLP(head = GaussianHead(), hidden = [64, 32, 16], epochs = 20)

models = [
    "MLP" => mlp_point,
    "MLP + nejistota" => mlp_gauss,
]

for (name, model) in models
    println("Trénuji: $name")
    fit!(model, train; val)
end

save_results(models, val, test, "mlp")
save_models("mlp"; point_model = mlp_point, gauss_model = mlp_gauss)

gauss_pred = predict(mlp_gauss, test)
lower, upper = interval(gauss_pred, INTERVAL)
pct = round(Int, 100 * INTERVAL)

fig = Figure(size = (1100, 450))

ax1 = Axis(fig[1, 1]; title = "Průběh tréninku", xlabel = "Epocha", ylabel = "Loss")
lines!(ax1, mlp_gauss.history; label = "train (NLL)")
lines!(ax1, mlp_gauss.val_history; label = "val (NLL)")
axislegend(ax1)

ax2 = Axis(fig[1, 2]; title = "Predikce s $(pct)% intervalem (posledních 200 událostí)",
    xlabel = "Pořadí události v testu", ylabel = "Magnitudo")
idx = max(1, length(test) - 199):length(test)
band!(ax2, idx, lower[idx], upper[idx]; color = (:darkorange, 0.25), label = "$(pct)% interval")
lines!(ax2, idx, point(gauss_pred)[idx]; color = :darkorange, label = "predikce")
scatter!(ax2, idx, test.y[idx]; color = :black, markersize = 5, label = "skutečnost")
axislegend(ax2; position = :lt)

mkpath(FIGURES_DIR)
figure_file = joinpath(FIGURES_DIR, "mlp.png")
save(figure_file, fig)