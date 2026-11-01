n_outputs(::PointHead) = 1
n_outputs(::GaussianHead) = 2

gaussian_params(out::AbstractMatrix) = (out[1, :], Flux.softplus.(out[2, :]))

"""
    loss(head, out, y)

Trénovací loss odpovídající hlavě: MSE pro bodový odhad, gaussovská
negativní log-věrohodnost pro normální rozdělení.
"""
loss(::PointHead, out::AbstractMatrix, y::AbstractVector) = mean(abs2.(vec(out) .- y))

function loss(::GaussianHead, out::AbstractMatrix, y::AbstractVector)
    μ, σ = gaussian_params(out)
    return mean(log.(σ) .+ abs2.((y .- μ) ./ σ) ./ 2)
end

"""
    to_prediction(head, out, y_mean, y_std)

Převede výstup sítě (ve standardizovaných jednotkách) na predikci v magnitudech.
"""
to_prediction(::PointHead, out, y_mean, y_std) =
    PointPrediction(Float64.(vec(out)) .* y_std .+ y_mean)

function to_prediction(::GaussianHead, out, y_mean, y_std)
    μ, σ = gaussian_params(out)
    return GaussianPrediction(Float64.(μ) .* y_std .+ y_mean, Float64.(σ) .* y_std)
end