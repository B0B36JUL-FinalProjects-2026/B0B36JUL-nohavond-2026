include("common.jl")
include("plots.jl")

Random.seed!(SEED)

train, val, test = prepare_data()

forest_point = RandomForest(n_trees = 100)
forest_samples = RandomForest(head = SampleHead(), n_trees = 100)

models = [
    "Random Forest" => forest_point,
    "Random Forest + nejistota" => forest_samples,
]

for (name, model) in models
    println("Trénuji: $name")
    fit!(model, train; val)
end

save_results(models, val, test, "random_forest")
save_models("random_forest"; point_model = forest_point, sample_model = forest_samples)

fig = Figure(size = (1100, 450))
ax = Axis(fig[1, 1]; title = "Random Forest: testovací data",
    xlabel = "Skutečné magnitudo", ylabel = "Predikované magnitudo", aspect = 1)
scatter!(ax, test.y, point(predict(forest_point, test)); markersize = 4, color = (:steelblue, 0.5))
ablines!(ax, 0, 1; color = :black, linestyle = :dash)
interval_axis!(fig[1, 2], predict(forest_samples, test), test.y; color = :steelblue)
save_figure(fig, "random_forest.png")
