"""
    LSTMModel(; head = PointHead(), hidden = 64, dense = 32, epochs = 30,
                batchsize = 256, learning_rate = 1e-3, seed = 42)

Rekurentní síť (Flux.jl) nad standardizovanými okny ve tvaru `5 × k × n`.
Skryté stavy LSTM ze všech časových kroků spojí [`TemporalAttention`](@ref),
výsledek projde plně propojenými vrstvami.

- `head = PointHead()`: jeden výstup, trénink přes MSE ([`PointPrediction`](@ref)),
- `head = GaussianHead()`: dva výstupy μ a σ, trénink přes NLL ([`GaussianPrediction`](@ref)).
"""
struct LSTMModel{H<:AbstractHead} <: NetworkModel
    head::H
    hidden::Int
    dense::Int
    epochs::Int
    batchsize::Int
    learning_rate::Float64
    seed::Int
    state::NetworkState
end

function LSTMModel(;
    head::AbstractHead = PointHead(),
    hidden::Integer = 64,
    dense::Integer = 32,
    epochs::Integer = 30,
    batchsize::Integer = 256,
    learning_rate::Real = 1e-3,
    seed::Integer = 42,
)
    return LSTMModel(head, hidden, dense, epochs, batchsize, learning_rate, seed, NetworkState())
end

function build_network(m::LSTMModel, n_in::Integer, n_out::Integer, rng::AbstractRNG)
    init = Flux.glorot_uniform(rng)
    return Flux.Chain(
        Flux.LSTM(n_in => m.hidden; init_kernel = init, init_recurrent_kernel = init),
        TemporalAttention(m.hidden; init),
        Flux.Dense(m.hidden => m.dense, Flux.relu; init),
        Flux.Dense(m.dense => n_out; init),
    )
end

network_inputs(m::LSTMModel, data::WindowedData) = Float32.(m.state.standardizer(data).X)

Base.show(io::IO, m::LSTMModel) =
    print(io, "LSTMModel(", m.hidden, " → ", m.dense, ", ", nameof(typeof(m.head)), ", ", network_state(m), ")")
