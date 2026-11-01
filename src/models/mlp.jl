"""
    MLP(; head = PointHead(), hidden = [64, 32], epochs = 50, batchsize = 256,
          learning_rate = 1e-3, seed = 42)

Vícevrstvý perceptron (Flux.jl) nad flattened a standardizovanými okny.

- `head = PointHead()`: jeden výstup, trénink přes MSE ([`PointPrediction`](@ref)),
- `head = GaussianHead()`: dva výstupy μ a σ, trénink přes NLL ([`GaussianPrediction`](@ref)).
"""
mutable struct MLP{H<:AbstractHead} <: AbstractQuakeModel
    head::H
    hidden::Vector{Int}
    epochs::Int
    batchsize::Int
    learning_rate::Float64
    seed::Int
    network::Any
    standardizer::Union{Nothing,Standardizer}
    y_mean::Float64
    y_std::Float64
    history::Vector{Float64}
    val_history::Vector{Float64}
end

function MLP(;
    head::AbstractHead = PointHead(),
    hidden::AbstractVector{<:Integer} = [64, 32],
    epochs::Integer = 50,
    batchsize::Integer = 256,
    learning_rate::Real = 1e-3,
    seed::Integer = 42,
)
    return MLP(head, collect(Int, hidden), epochs, batchsize, learning_rate, seed,
        nothing, nothing, NaN, NaN, Float64[], Float64[])
end

function build_mlp(n_in::Integer, hidden::Vector{Int}, n_out::Integer, rng::AbstractRNG)
    init = Flux.glorot_uniform(rng)
    dims = [n_in; hidden]
    layers = Any[Flux.Dense(a => b, Flux.relu; init) for (a, b) in zip(dims[1:end-1], dims[2:end])]
    push!(layers, Flux.Dense(dims[end] => n_out; init))
    return Flux.Chain(layers...)
end

mlp_inputs(m::MLP, data::WindowedData) = Float32.(flatten_windows(m.standardizer(data)))
mlp_targets(m::MLP, data::WindowedData) = Float32.((data.y .- m.y_mean) ./ m.y_std)

function fit!(m::MLP, train::WindowedData; val::Union{Nothing,WindowedData} = nothing)
    rng = Xoshiro(m.seed)
    m.standardizer = fit_standardizer(train)
    m.y_mean = mean(train.y)
    m.y_std = std(train.y)

    X = mlp_inputs(m, train)
    m.network = build_mlp(size(X, 1), m.hidden, n_outputs(m.head), rng)

    val_data = val === nothing ? nothing : (mlp_inputs(m, val), mlp_targets(m, val))
    m.history, m.val_history = train_network!(m.network, m.head, (X, mlp_targets(m, train)), val_data;
        epochs = m.epochs, batchsize = m.batchsize, learning_rate = m.learning_rate, rng)
    return m
end

function predict(m::MLP, data::WindowedData)
    m.network === nothing && error("Model ještě není natrénovaný, zavolej nejdřív fit!(model, train).")
    out = m.network(mlp_inputs(m, data))
    return to_prediction(m.head, out, m.y_mean, m.y_std)
end

function Base.show(io::IO, m::MLP)
    state = m.network === nothing ? "nenatrénovaný" : "natrénovaný"
    print(io, "MLP(", join(m.hidden, "→"), ", ", nameof(typeof(m.head)), ", ", state, ")")
end