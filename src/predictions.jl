abstract type AbstractPrediction end
struct PointPrediction <: AbstractPrediction
    ŷ::Vector{Float64}
end
struct SamplePrediction <: AbstractPrediction
    samples::Matrix{Float64}
end

Base.length(p::PointPrediction) = length(p.ŷ)
Base.length(p::SamplePrediction) = size(p.samples, 2)

"""
    point(p::AbstractPrediction)

Bodový odhad z libovolné predikce. U vzorků je to jejich průměr.
"""
point(p::PointPrediction) = p.ŷ
point(p::SamplePrediction) = vec(mean(p.samples; dims = 1))

"""
    interval(p::SamplePrediction, level = 0.9) -> (dolní, horní)

COterval s pravděpodobností `level` z kvantilů vzorků.
"""
function interval(p::SamplePrediction, level::Real = 0.9)
    α = (1 - level) / 2
    lower = [quantile(col, α) for col in eachcol(p.samples)]
    upper = [quantile(col, 1 - α) for col in eachcol(p.samples)]
    return lower, upper
end