include("common.jl")

Random.seed!(SEED)

train, val, test = prepare_data()

baseline_point = MeanBaseline()
baseline_samples = MeanBaseline(head = SampleHead())

models = [
    "Baseline (průměr)" => baseline_point,
    "Baseline (rozdělení)" => baseline_samples,
]

for (name, model) in models
    println("Trénuji: $name")
    fit!(model, train)
end

save_results(models, val, test, "baseline")
save_models("baseline"; point_model = baseline_point, sample_model = baseline_samples)