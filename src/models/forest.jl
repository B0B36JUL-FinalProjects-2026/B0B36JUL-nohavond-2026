"""
    RandomForest(; head = PointHead(), n_trees = 100, max_depth = -1,
                   min_samples_leaf = 5, partial_sampling = 0.7, seed = 42)

- `head = PointHead()`: predikce je průměr stromů ([`PointPrediction`](@ref)),
- `head = SampleHead()`: predikce jednotlivých stromů tvoří vzorky
  ([`SamplePrediction`](@ref)), jejich rozptyl je odhad nejistoty.
"""
mutable struct RandomForest{H<:AbstractHead} <: AbstractQuakeModel
    head::H
    n_trees::Int
    max_depth::Int
    min_samples_leaf::Int
    partial_sampling::Float64
    seed::Int
    forest::Union{Nothing,DecisionTree.Ensemble}
end

function RandomForest(;
    head::AbstractHead = PointHead(),
    n_trees::Integer = 100,
    max_depth::Integer = -1,
    min_samples_leaf::Integer = 5,
    partial_sampling::Real = 0.7,
    seed::Integer = 42,
)
    return RandomForest(head, n_trees, max_depth, min_samples_leaf, partial_sampling, seed, nothing)
end

# DecisionTree.jl čeká vzorky v řádcích, okna mají vzorky ve sloupcích.
forest_features(data::WindowedData) = permutedims(flatten_windows(data))

function fit!(m::RandomForest, train::WindowedData)
    X = forest_features(train)
    n_subfeatures = max(1, size(X, 2) ÷ 3)
    m.forest = DecisionTree.build_forest(
        train.y, X, n_subfeatures, m.n_trees, m.partial_sampling, m.max_depth, m.min_samples_leaf;
        rng = m.seed,
    )
    return m
end

"""
    tree_predictions(m::RandomForest, data) -> Matrix{Float64}

Predikce každého stromu zvlášť, tvar `počet stromů × počet oken`.
"""
function tree_predictions(m::RandomForest, data::WindowedData)
    m.forest === nothing && error("Model ještě není natrénovaný, zavolej nejdřív fit!(model, train).")
    X = forest_features(data)
    P = Matrix{Float64}(undef, length(m.forest.trees), length(data))
    for (t, tree) in enumerate(m.forest.trees)
        P[t, :] = DecisionTree.apply_tree(tree, X)
    end
    return P
end

predict(m::RandomForest, data::WindowedData) = aggregate(m.head, tree_predictions(m, data))

aggregate(::PointHead, P::Matrix{Float64}) = PointPrediction(vec(mean(P; dims = 1)))
aggregate(::SampleHead, P::Matrix{Float64}) = SamplePrediction(P)

function Base.show(io::IO, m::RandomForest)
    state = m.forest === nothing ? "nenatrénovaný" : "natrénovaný"
    print(io, "RandomForest(", m.n_trees, " stromů, ", nameof(typeof(m.head)), ", ", state, ")")
end