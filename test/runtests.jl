using EarthquakePredictionJapan
using Test
using Aqua
using Dates

@testset "EarthquakePredictionJapan" begin
    include("test_data.jl")

    @testset "Aqua" begin
        Aqua.test_all(EarthquakePredictionJapan; stale_deps = false)
    end
end