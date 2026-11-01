# Testy nacitani dat na souboru vygenerovanem v Claude.
# Soubor mam pouze pro testovani!

const SAMPLE_CATALOG = joinpath(@__DIR__, "data", "sample_usgs.csv")

@testset "Data" begin
    @testset "GeoBounds" begin
        @test (35.7, 139.7) in JAPAN          # Tokio
        @test !((10.0, 139.0) in JAPAN)       # příliš na jih
        @test !((35.0, 100.0) in JAPAN)       # příliš na západ
        @test (24.0, 122.0) in JAPAN          # hranice patří dovnitř
    end

    @testset "parse_usgs_time" begin
        parse_time = EarthquakePredictionJapan.parse_usgs_time
        expected = DateTime(2011, 3, 11, 5, 46, 24, 120)
        @test parse_time("2011-03-11T05:46:24.120Z") == expected
        @test parse_time(expected) == expected
    end

    @testset "load_catalog" begin
        df = load_catalog(SAMPLE_CATALOG)

        @test names(df) == string.(EarthquakePredictionJapan.CATALOG_COLUMNS)
        @test df.id == ["test_b", "test_a"]
        @test issorted(df.time)
        @test eltype(df.time) == DateTime
        @test df.time[1] == DateTime(2011, 3, 11, 5, 46, 24, 120)
        @test all(>(5), df.magnitude)
        @test all((lat, lon) in JAPAN for (lat, lon) in zip(df.latitude, df.longitude))
        @test eltype(df.depth) == Float64
    end

    @testset "load_catalog s nižším prahem" begin
        df = load_catalog(SAMPLE_CATALOG; min_magnitude = 4.5)
        @test df.id == ["test_b", "test_c", "test_d", "test_a"]
    end

    @testset "chybějící soubor" begin
        @test_throws ErrorException load_catalog("neexistujici_soubor.csv")
    end

    @testset "Split" begin
    df = load_catalog(SAMPLE_CATALOG; min_magnitude = 4.5)
    s = chronological_split(df; train = 0.5, val = 0.25)
    @test size.((s.train, s.val, s.test), 1) == (2, 1, 1)
    @test maximum(s.train.time) < minimum(s.val.time) < minimum(s.test.time)
    @test_throws ArgumentError chronological_split(df; train = 0.9, val = 0.2)
    end



end