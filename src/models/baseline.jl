"""
    MeanBaseline()

Referenční model: vždy predikuje průměrné magnitudo z trénovacích dat.
Každý další model by ho měl porazit.
"""
mutable struct MeanBaseline <: AbstractQuakeModel
    μ::Float64
end
MeanBaseline() = MeanBaseline(NaN)

function fit!(m::MeanBaseline, train::WindowedData)
    m.μ = mean(train.y)
    return m
end

predict(m::MeanBaseline, data::WindowedData) = PointPrediction(fill(m.μ, length(data)))

Base.show(io::IO, m::MeanBaseline) = print(io, "MeanBaseline(μ = ", round(m.μ; digits = 3), ")")