"""
    NetworkState()

Naučený stav modelu z neuronové sítě: síť, standardizace vstupů, škálování
cíle a průběh tréninku. Před voláním [`fit!`](@ref) je prázdný.
"""
mutable struct NetworkState
    network::Any
    n_in::Int
    standardizer::Union{Nothing,Standardizer}
    y_mean::Float64
    y_std::Float64
    history::Vector{Float64}
    val_history::Vector{Float64}
end

NetworkState() = NetworkState(nothing, 0, nothing, NaN, NaN, Float64[], Float64[])

"""
Společný nadtyp modelů z neuronových sítí Flux.jl. Trénink a predikce jsou
společné, konkrétní síť dodává jen:

- `network_inputs(m, data)` – vstup sítě ze standardizovaných oken,
- `build_network(m, n_in, n_out, rng)` – síť s `n_in` vstupy a `n_out` výstupy.

Podtypy musí mít pole `head, epochs, batchsize, learning_rate, seed`
a `state::NetworkState`.
"""
abstract type NetworkModel <: AbstractQuakeModel end

function network_inputs end
function build_network end

network_targets(m::NetworkModel, data::WindowedData) =
    Float32.((data.y .- m.state.y_mean) ./ m.state.y_std)

function fit!(m::NetworkModel, train::WindowedData; val::Union{Nothing,WindowedData} = nothing)
    rng = Xoshiro(m.seed)
    s = m.state
    s.standardizer = fit_standardizer(train)
    s.y_mean = mean(train.y)
    s.y_std = std(train.y)

    X = network_inputs(m, train)
    s.n_in = size(X, 1)
    s.network = build_network(m, s.n_in, n_outputs(m.head), rng)

    val_data = val === nothing ? nothing : (network_inputs(m, val), network_targets(m, val))
    s.history, s.val_history = train_network!(s.network, m.head, (X, network_targets(m, train)), val_data;
        epochs = m.epochs, batchsize = m.batchsize, learning_rate = m.learning_rate, rng)
    return m
end

function predict(m::NetworkModel, data::WindowedData)
    m.state.network === nothing && error("Model ještě není natrénovaný, zavolej nejdřív fit!(model, train).")
    out = m.state.network(network_inputs(m, data))
    return to_prediction(m.head, out, m.state.y_mean, m.state.y_std)
end

network_state(m::NetworkModel) = m.state.network === nothing ? "nenatrénovaný" : "natrénovaný"

"""
    model_state(m::NetworkModel) -> NamedTuple

Vše, co je potřeba k obnovení natrénovaného modelu, jen z obyčejných polí
a čísel: `Flux.state` sítě, počet vstupů, parametry standardizace a škálování
cíle. Na rozdíl od uložení celé struktury modelu se uložený soubor nerozbije
při změně definic typů v balíčku.
"""
function model_state(m::NetworkModel)
    s = m.state
    s.network === nothing && error("Model ještě není natrénovaný, není co ukládat.")
    return (;
        network = Flux.state(s.network),
        n_in = s.n_in,
        μ = s.standardizer.μ,
        σ = s.standardizer.σ,
        y_mean = s.y_mean,
        y_std = s.y_std,
        history = s.history,
        val_history = s.val_history,
    )
end

"""
    load_state!(m::NetworkModel, saved) -> m

Obnoví model ze stavu vráceného [`model_state`](@ref). Model musí mít stejné
hyperparametry (architekturu a hlavu) jako ten, ze kterého stav pochází.
"""
function load_state!(m::NetworkModel, saved::NamedTuple)
    s = m.state
    s.n_in = saved.n_in
    s.standardizer = Standardizer(saved.μ, saved.σ)
    s.y_mean, s.y_std = saved.y_mean, saved.y_std
    s.history, s.val_history = saved.history, saved.val_history
    s.network = build_network(m, s.n_in, n_outputs(m.head), Xoshiro(m.seed))
    Flux.loadmodel!(s.network, saved.network)
    return m
end
