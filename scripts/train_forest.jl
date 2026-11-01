include("common.jl")
using CairoMakie
using JLD2

Random.seed!(SEED)

train, val, test = prepare_data()

models = [
    "Baseline (průměr)" => MeanBaseline(),
    "Random Forest" => RandomForest(n_trees = 200),
    "Random Forest + nejistota" => RandomForest(head = SampleHead(), n_trees = 200),
]

for (name, model) in models
    println("Trénuji: $name")
    fit!(model, train)
end

save_results(models, val, test, "random_forest")

mkpath(MODELS_DIR)
model_file = joinpath(MODELS_DIR, "random_forest.jld2")
jldsave(model_file; point_model = models[2].second, sample_model = models[3].second, k = K)

point_pred = predict(models[2].second, test)
sample_pred = predict(models[3].second, test)
lower, upper = interval(sample_pred, 0.9)

fig = Figure(size = (1100, 450))

ax1 = Axis(fig[1, 1]; title = "Random Forest: testovací data",
    xlabel = "Skutečné magnitude", ylabel = "Predikované magnitude", aspect = 1)
scatter!(ax1, test.y, point(point_pred); markersize = 4, color = (:steelblue, 0.5))
ablines!(ax1, 0, 1; color = :black, linestyle = :dash)

ax2 = Axis(fig[1, 2]; title = "Predikce s 90% intervalem (posledních 200 událostí)",
    xlabel = "Pořadí události v testu", ylabel = "Magnitude")
idx = max(1, length(test) - 199):length(test)
band!(ax2, idx, lower[idx], upper[idx]; color = (:steelblue, 0.25), label = "90% interval")
lines!(ax2, idx, point(sample_pred)[idx]; color = :steelblue, label = "predikce")
scatter!(ax2, idx, test.y[idx]; color = :black, markersize = 5, label = "skutečnost")
axislegend(ax2; position = :lt)

mkpath(FIGURES_DIR)
figure_file = joinpath(FIGURES_DIR, "random_forest.png")
save(figure_file, fig)