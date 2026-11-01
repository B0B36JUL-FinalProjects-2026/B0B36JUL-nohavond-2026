include("common.jl")
using CairoMakie
using JLD2

Random.seed!(SEED)

train, val, test = prepare_data()

forest_point = RandomForest(n_trees = 50)
forest_samples = RandomForest(head = SampleHead(), n_trees = 50)

models = [
    "Baseline (průměr)" => MeanBaseline(),
    "Baseline (rozdělení)" => MeanBaseline(head = SampleHead()),
    "Random Forest" => forest_point,
    "Random Forest + nejistota" => forest_samples,
]

for (name, model) in models
    fit!(model, train)
end

save_results(models, val, test, "random_forest")

mkpath(MODELS_DIR)
model_file = joinpath(MODELS_DIR, "random_forest.jld2")
jldsave(model_file; point_model = forest_point, sample_model = forest_samples, k = K)

point_pred = predict(forest_point, test)
sample_pred = predict(forest_samples, test)
lower, upper = interval(sample_pred, INTERVAL)

fig = Figure(size = (1100, 450))

ax1 = Axis(fig[1, 1]; title = "Random Forest: testovací data",
    xlabel = "Skutečné magnitudo", ylabel = "Predikované magnitudo", aspect = 1)
scatter!(ax1, test.y, point(point_pred); markersize = 4, color = (:steelblue, 0.5))
ablines!(ax1, 0, 1; color = :black, linestyle = :dash)

ax2 = Axis(fig[1, 2]; title = "Predikce s $(round(Int, INTERVAL * 100)) intervalem (posledních 200 událostí)",
    xlabel = "Pořadí události v testu", ylabel = "Magnitudo")
idx = max(1, length(test) - 199):length(test)
band!(ax2, idx, lower[idx], upper[idx]; color = (:steelblue, 0.25), label = "$(round(Int, INTERVAL * 100))% interval")
lines!(ax2, idx, point(sample_pred)[idx]; color = :steelblue, label = "predikce")
scatter!(ax2, idx, test.y[idx]; color = :black, markersize = 5, label = "skutečnost")
axislegend(ax2; position = :lt)

mkpath(FIGURES_DIR)
figure_file = joinpath(FIGURES_DIR, "random_forest.png")
save(figure_file, fig)