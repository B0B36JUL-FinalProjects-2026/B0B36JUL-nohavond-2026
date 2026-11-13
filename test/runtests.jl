using EarthquakePredictionJapan
using Test
using Aqua
using Dates

@testset "EarthquakePredictionJapan" begin
    include("test_data.jl")
    include("test_models.jl")

    @testset "Aqua" begin
        Aqua.test_all(EarthquakePredictionJapan)
    end
end