"""
    MLP(; head = PointHead(), hidden = [64, 32], epochs = 50, batchsize = 256,
          learning_rate = 1e-3, seed = 42)

Vícevrstvý perceptron (Flux.jl) nad flattened a standardizovanými okny.

- `head = PointHead()`: jeden výstup, trénink přes MSE ([`PointPrediction`](@ref)),
- `head = GaussianHead()`: dva výstupy μ a σ, trénink přes NLL ([`GaussianPrediction`](@ref)).
"""
mutable struct MLP{H<:AbstractHead} <: NetworkModel
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

function build_network(m::MLP, X::AbstractMatrix, n_out::Integer, rng::AbstractRNG)
    init = Flux.glorot_uniform(rng)
    dims = [size(X, 1); m.hidden]
    layers = Any[Flux.Dense(a => b, Flux.relu; init) for (a, b) in zip(dims[1:end-1], dims[2:end])]
    push!(layers, Flux.Dense(dims[end] => n_out; init))
    return Flux.Chain(layers...)
end

network_inputs(m::MLP, data::WindowedData) = Float32.(flatten_windows(m.standardizer(data)))

Base.show(io::IO, m::MLP) =
    print(io, "MLP(", join(m.hidden, "→"), ", ", nameof(typeof(m.head)), ", ", network_state(m), ")")