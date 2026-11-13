include("common.jl")
include("plots.jl")

Random.seed!(SEED)

train, val, test = prepare_data()

mlp_point = MLP(hidden = [64, 32, 16], epochs = 30)
mlp_gauss = MLP(head = GaussianHead(), hidden = [64, 32, 16], epochs = 30)

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
network_figure(mlp_gauss, test, "mlp.png")
