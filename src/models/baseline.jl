"""
    MeanBaseline(; head = PointHead(), n_samples = 200)

Referenční model, který predikuje jen ze známého rozdělení
magnitud v trénovacích datech.
"""
mutable struct MeanBaseline{H<:AbstractHead} <: AbstractQuakeModel
    head::H
    n_samples::Int
    μ::Float64
    quantiles::Vector{Float64}
end

MeanBaseline(; head::AbstractHead = PointHead(), n_samples::Integer = 200) =
    MeanBaseline(head, n_samples, NaN, Float64[])

function fit!(m::MeanBaseline, train::WindowedData)
    m.μ = mean(train.y)
    m.quantiles = quantile(train.y, (1:m.n_samples) ./ (m.n_samples + 1))
    return m
end

predict(m::MeanBaseline{PointHead}, data::WindowedData) =
    PointPrediction(fill(m.μ, length(data)))

predict(m::MeanBaseline{SampleHead}, data::WindowedData) =
    SamplePrediction(repeat(m.quantiles, 1, length(data)))

Base.show(io::IO, m::MeanBaseline) =
    print(io, "MeanBaseline(", nameof(typeof(m.head)), ", μ = ", round(m.μ; digits = 3), ")")