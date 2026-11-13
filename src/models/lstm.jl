"""
    LSTMModel(; head = PointHead(), hidden = 64, dense = 32, epochs = 50,
                batchsize = 256, learning_rate = 1e-3, seed = 42)

Rekurentní síť (Flux.jl) nad standardizovanými okny ve tvaru `5 × k × n`.
Skryté stavy LSTM ze všech časových kroků spojí [`TemporalAttention`](@ref),
výsledek projde plně propojenými vrstvami.

- `head = PointHead()`: jeden výstup, trénink přes MSE ([`PointPrediction`](@ref)),
- `head = GaussianHead()`: dva výstupy μ a σ, trénink přes NLL ([`GaussianPrediction`](@ref)).
"""
mutable struct LSTMModel{H<:AbstractHead} <: NetworkModel
    head::H
    hidden::Int
    dense::Int
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

function LSTMModel(;
    head::AbstractHead = PointHead(),
    hidden::Integer = 64,
    dense::Integer = 32,
    epochs::Integer = 50,
    batchsize::Integer = 256,
    learning_rate::Real = 1e-3,
    seed::Integer = 42,
)
    return LSTMModel(head, hidden, dense, epochs, batchsize, learning_rate, seed,
        nothing, nothing, NaN, NaN, Float64[], Float64[])
end

function build_network(m::LSTMModel, X::AbstractArray{<:Real,3}, n_out::Integer, rng::AbstractRNG)
    init = Flux.glorot_uniform(rng)
    return Flux.Chain(
        Flux.LSTM(size(X, 1) => m.hidden; init_kernel = init, init_recurrent_kernel = init),
        TemporalAttention(m.hidden; init),
        Flux.Dense(m.hidden => m.dense, Flux.relu; init),
        Flux.Dense(m.dense => n_out; init),
    )
end

network_inputs(m::LSTMModel, data::WindowedData) = Float32.(m.standardizer(data).X)

Base.show(io::IO, m::LSTMModel) =
    print(io, "LSTMModel(", m.hidden, " → ", m.dense, ", ", nameof(typeof(m.head)), ", ", network_state(m), ")")
