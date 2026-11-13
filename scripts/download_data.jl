# Stáhne katalog zemětřesení USGS pro oblast Japonska.
# Spouští se z kořene repozitáře: julia --project=. scripts/download_data.jl

using Downloads
using CSV
using DataFrames
using EarthquakePredictionJapan: JAPAN, default_catalog_path

const API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Stahuje se se slabší hranicí, než jakou používá load_catalog (> 5.0),
# aby šlo s hranicí experimentovat bez nového stahování.
const MIN_MAGNITUDE = 4.5

# Pevné období (celé roky), aby každé stažení dávalo stejná data.
const FIRST_YEAR = 1990
const LAST_YEAR = 2025

const OUTPUT_FILE = default_catalog_path()

"""
    query_url(year)

Sestaví URL dotazu na USGS API pro jeden kalendářní rok.
"""
function query_url(year::Integer)
    params = [
        "format" => "csv",
        "starttime" => "$(year)-01-01",
        "endtime" => "$(year + 1)-01-01",
        "minlatitude" => JAPAN.min_lat,
        "maxlatitude" => JAPAN.max_lat,
        "minlongitude" => JAPAN.min_lon,
        "maxlongitude" => JAPAN.max_lon,
        "minmagnitude" => MIN_MAGNITUDE,
        "eventtype" => "earthquake",
        "orderby" => "time-asc",
    ]
    return API_URL * "?" * join(["$k=$v" for (k, v) in params], "&")
end

"""
    download_year(year, dir)

Stáhne jeden rok do dočasné složky `dir` a vrátí ho jako DataFrame.
"""
function download_year(year::Integer, dir::AbstractString)
    print("  $year: stahuji... ")
    file = Downloads.download(query_url(year), joinpath(dir, "$(year).csv"))
    df = CSV.read(file, DataFrame)
    println("$(nrow(df)) událostí")
    nrow(df) >= 20_000 && @warn "Rok $year dosáhl limitu 20 000 událostí, data mohou být neúplná."
    sleep(1)
    return df
end

function main()
    tables = mktempdir() do dir
        [download_year(y, dir) for y in FIRST_YEAR:LAST_YEAR]
    end

    df = reduce((a, b) -> vcat(a, b; cols = :union), tables)

    mkpath(dirname(OUTPUT_FILE))
    CSV.write(OUTPUT_FILE, df)
    println("Uloženo $(nrow(df)) událostí do $OUTPUT_FILE")

    return df
end

main()