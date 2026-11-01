# Stáhne katalog zemětřesení USGS pro oblast Japonska

using Downloads
using Dates
using CSV
using DataFrames

const API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Výřez kolem Japonska
const MIN_LAT, MAX_LAT = 24, 46
const MIN_LON, MAX_LON = 122, 148

const MIN_MAGNITUDE = 4.5

const FIRST_YEAR = 1990
const LAST_YEAR = year(today())

const OUTPUT_FILE = joinpath(@__DIR__, "..", "data", "japan_usgs.csv")

"""
    query_url(year)

Sestaví URL dotazu na USGS API pro jeden kalendářní rok.
"""
function query_url(year::Integer)
    params = [
        "format" => "csv",
        "starttime" => "$(year)-01-01",
        "endtime" => "$(year + 1)-01-01",
        "minlatitude" => MIN_LAT,
        "maxlatitude" => MAX_LAT,
        "minlongitude" => MIN_LON,
        "maxlongitude" => MAX_LON,
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

    return df
end

main()