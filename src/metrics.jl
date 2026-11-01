abstract type AbstractMetric end

struct MAE <: AbstractMetric end
struct RMSE <: AbstractMetric end
struct CRPS <: AbstractMetric end
struct Coverage <: AbstractMetric
    level::Float64
end
Coverage() = Coverage(0.9)

Base.show(io::IO, ::MAE) = print(io, "MAE")
Base.show(io::IO, ::RMSE) = print(io, "RMSE")
Base.show(io::IO, ::CRPS) = print(io, "CRPS")
Base.show(io::IO, m::Coverage) = print(io, "Pokrytí ", round(Int, 100 * m.level), " %")

"""
    evaluate(metric, prediction, y)

Spočítá metriku pro predikce a skutečné hodnoty `y`. Metoda se vybírá podle typu
metriky i typu predikce. Pokud metrika pro daný typ predikce nedává smysl, vrací
`missing`.
"""
function evaluate end

evaluate(::MAE, p::AbstractPrediction, y::AbstractVector) = mean(abs.(point(p) .- y))
evaluate(::RMSE, p::AbstractPrediction, y::AbstractVector) = sqrt(mean((point(p) .- y) .^ 2))
evaluate(::CRPS, p::PointPrediction, y::AbstractVector) = evaluate(MAE(), p, y)

function evaluate(::CRPS, p::SamplePrediction, y::AbstractVector)
    total = 0.0
    for (j, yj) in enumerate(y)
        s = view(p.samples, :, j)
        total += mean(abs.(s .- yj)) - 0.5 * mean(abs.(s .- s'))
    end
    return total / length(y)
end

evaluate(::Coverage, ::PointPrediction, ::AbstractVector) = missing

function evaluate(m::Coverage, p::SamplePrediction, y::AbstractVector)
    lower, upper = interval(p, m.level)
    return mean(lower .<= y .<= upper)
end