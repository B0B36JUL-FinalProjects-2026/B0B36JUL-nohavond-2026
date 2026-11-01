"""
    Standardizer(μ, σ)

Standardizace příznaků na nulový průměr a jednotkový rozptyl.
"""
struct Standardizer
    μ::Array{Float64,3}
    σ::Array{Float64,3}
end

"""
    fit_standardizer(w::WindowedData) -> Standardizer

Spočítá průměr a směrodatnou odchylku každého příznaku přes všechna okna
a časové kroky.
"""
function fit_standardizer(w::WindowedData)
    μ = mean(w.X; dims = (2, 3))
    σ = std(w.X; dims = (2, 3))
    σ[σ .== 0] .= 1
    return Standardizer(μ, σ)
end

(s::Standardizer)(w::WindowedData) =
    WindowedData((w.X .- s.μ) ./ s.σ, w.y, w.time, w.latitude, w.longitude)