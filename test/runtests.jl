using EarthquakePredictionJapan
using Test
using Aqua

@testset "EarthquakePredictionJapan" begin
    @testset "Aqua" begin
        Aqua.test_all(EarthquakePredictionJapan; stale_deps = false)
    end
end