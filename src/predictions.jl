abstract type AbstractPrediction end

abstract type ProbabilisticPrediction <: AbstractPrediction end
struct PointPrediction <: AbstractPrediction
    ŷ::Vector{Float64}
end
struct SamplePrediction <: ProbabilisticPrediction
    samples::Matrix{Float64}
end
struct GaussianPrediction <: ProbabilisticPrediction
    μ::Vector{Float64}
    σ::Vector{Float64}
end

Base.length(p::PointPrediction) = length(p.ŷ)
Base.length(p::SamplePrediction) = size(p.samples, 2)
Base.length(p::GaussianPrediction) = length(p.μ)


point(p::PointPrediction) = p.ŷ
point(p::SamplePrediction) = vec(mean(p.samples; dims = 1))
point(p::GaussianPrediction) = p.μ

"""
    interval(p::ProbabilisticPrediction, level = 0.9) -> (dolní, horní)

Centrální predikční interval s pravděpodobností `level`.
"""
function interval(p::SamplePrediction, level::Real = 0.9)
    α = (1 - level) / 2
    lower = [quantile(col, α) for col in eachcol(p.samples)]
    upper = [quantile(col, 1 - α) for col in eachcol(p.samples)]
    return lower, upper
end

function interval(p::GaussianPrediction, level::Real = 0.9)
    z = quantile(Normal(), (1 + level) / 2)
    return p.μ .- z .* p.σ, p.μ .+ z .* p.σ
end