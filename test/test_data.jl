# Testy nacitani dat na souboru vygenerovanem v Claude.
# Soubor mam pouze pro testovani!

const SAMPLE = joinpath(@__DIR__, "data", "sample_usgs.csv")

@testset "Data" begin
    @test (35.7, 139.7) in JAPAN
    @test !((10.0, 139.0) in JAPAN)

    df = load_catalog(SAMPLE)
    @test df.id == ["test_b", "test_a"]
    @test issorted(df.time)
    @test df.time[1] == DateTime(2011, 3, 11, 5, 46, 24, 120)
    @test all(>(5), df.magnitude)

    @test load_catalog(SAMPLE; min_magnitude = 4.5).id == ["test_b", "test_c", "test_d", "test_a"]
    @test_throws ErrorException load_catalog("a_co_kdyz_tu_je_neexistuje.csv")
end

@testset "Split" begin
    df = load_catalog(SAMPLE; min_magnitude = 4.5)
    s = chronological_split(df; train = 0.5, val = 0.25)
    @test size.((s.train, s.val, s.test), 1) == (2, 1, 1)
    @test maximum(s.train.time) < minimum(s.val.time) < minimum(s.test.time)
    @test_throws ArgumentError chronological_split(df; train = 0.9, val = 0.2)
end

@testset "Windows" begin
    df = load_catalog(SAMPLE; min_magnitude = 4.5)
    w = make_windows(df; k = 2)
    @test size(w.X) == (5, 2, 2)
    @test w.y == df.magnitude[3:4]
    @test w.X[5, :, 1] == df.magnitude[1:2]
    @test size(flatten_windows(w)) == (10, 2)
    @test_throws ArgumentError make_windows(df; k = 4)

    z = fit_standardizer(w)(w)
    @test isapprox(sum(z.X; dims = (2, 3)), zeros(5, 1, 1); atol = 1e-10)
    @test z.y == w.y
end