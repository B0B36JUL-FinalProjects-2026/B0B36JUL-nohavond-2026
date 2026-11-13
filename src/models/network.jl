"""
Společný nadtyp modelů z neuronových sítí Flux.jl. Trénink a predikce jsou
společné, konkrétní síť dodává jen:

- `network_inputs(m, data)` – vstup sítě ze standardizovaných oken,
- `build_network(m, X, n_out, rng)` – síť pro vstup `X` s `n_out` výstupy.

Podtypy musí mít pole `head, epochs, batchsize, learning_rate, seed, network,
standardizer, y_mean, y_std, history, val_history`.
"""
abstract type NetworkModel <: AbstractQuakeModel end

function network_inputs end
function build_network end

network_targets(m::NetworkModel, data::WindowedData) = Float32.((data.y .- m.y_mean) ./ m.y_std)

function fit!(m::NetworkModel, train::WindowedData; val::Union{Nothing,WindowedData} = nothing)
    rng = Xoshiro(m.seed)
    m.standardizer = fit_standardizer(train)
    m.y_mean = mean(train.y)
    m.y_std = std(train.y)

    X = network_inputs(m, train)
    m.network = build_network(m, X, n_outputs(m.head), rng)

    val_data = val === nothing ? nothing : (network_inputs(m, val), network_targets(m, val))
    m.history, m.val_history = train_network!(m.network, m.head, (X, network_targets(m, train)), val_data;
        epochs = m.epochs, batchsize = m.batchsize, learning_rate = m.learning_rate, rng)
    return m
end

function predict(m::NetworkModel, data::WindowedData)
    m.network === nothing && error("Model ještě není natrénovaný, zavolej nejdřív fit!(model, train).")
    out = m.network(network_inputs(m, data))
    return to_prediction(m.head, out, m.y_mean, m.y_std)
end

network_state(m::NetworkModel) = m.network === nothing ? "nenatrénovaný" : "natrénovaný"
