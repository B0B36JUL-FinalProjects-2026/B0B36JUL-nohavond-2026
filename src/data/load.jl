# Načtení a vyčištění katalogu zemětřesení USGS.

"""
    GeoBounds(min_lat, max_lat, min_lon, max_lon)

Obdélníková zeměpisná oblast ve stupních.
Japonsko v datech je vyfiltrované, ale kdybych chtěl experimentovat na jiných datech.
"""
struct GeoBounds
    min_lat::Float64
    max_lat::Float64
    min_lon::Float64
    max_lon::Float64
end

Base.in((lat, lon)::Tuple{Real,Real}, b::GeoBounds) =
    b.min_lat <= lat <= b.max_lat && b.min_lon <= lon <= b.max_lon

"""
    JAPAN

Výřez kolem Japonska.
"""
const JAPAN = GeoBounds(24, 46, 122, 148)

"""Sloupce, které vrací [`load_catalog`](@ref), v tomto pořadí."""
const CATALOG_COLUMNS = [:id, :time, :latitude, :longitude, :depth, :magnitude]

"""
    default_catalog_path()

Cesta k souboru `data/japan_usgs.csv` v kořeni balíčku.
"""
default_catalog_path() = joinpath(pkgdir(@__MODULE__), "data", "japan_usgs.csv")

# USGS ukládá čas jako "2011-03-11T05:46:24.120Z" (UTC). Pokud už čas
# parsoval CSV.jl, přijde rovnou DateTime. Multiple dispatch pokryje oba případy.
parse_usgs_time(t::DateTime) = t
parse_usgs_time(s::AbstractString) = DateTime(rstrip(s, 'Z'))

"""
    load_catalog(path = default_catalog_path(); min_magnitude = 5.0, bounds = JAPAN)

Načte katalog zemětřesení USGS z CSV a vyčistí ho funkcí [`clean_catalog`](@ref).
Data se stahují skriptem `scripts/download_data.jl`.
"""
function load_catalog(
    path::AbstractString = default_catalog_path();
    min_magnitude::Real = 5.0,
    bounds::GeoBounds = JAPAN,
)
    isfile(path) || error(
        "Soubor $path neexistuje. Data stáhneš příkazem " *
        "`julia --project=. scripts/download_data.jl`.",
    )
    raw = CSV.read(path, DataFrame)
    return clean_catalog(raw; min_magnitude, bounds)
end

"""
    clean_catalog(raw::DataFrame; min_magnitude = 5.0, bounds = JAPAN)

Vyčistí surovou tabulku z USGS:

1. ponechá jen události typu `"earthquake"`,
2. vybere sloupce [`CATALOG_COLUMNS`](@ref) a přejmenuje `mag` na `magnitude`,
3. odstraní řádky s chybějícími hodnotami a duplicitní `id`,
4. ponechá jen události s magnitudem ostře větším než `min_magnitude`,
5. ponechá jen události uvnitř `bounds`,
6. převede čas na `DateTime` a seřadí tabulku podle času.
"""
function clean_catalog(raw::DataFrame; min_magnitude::Real = 5.0, bounds::GeoBounds = JAPAN)
    df = rename(raw, :mag => :magnitude)

    if hasproperty(df, :type)
        df = filter(:type => ==("earthquake"), df)
    end

    df = select(df, CATALOG_COLUMNS)
    dropmissing!(df)
    unique!(df, :id)

    filter!(:magnitude => >(min_magnitude), df)
    filter!([:latitude, :longitude] => (lat, lon) -> (lat, lon) in bounds, df)

    # vsechno ve float 
    df.time = parse_usgs_time.(df.time)
    for col in (:latitude, :longitude, :depth, :magnitude)
        df[!, col] = Float64.(df[!, col])
    end

    sort!(df, :time)
    return df
end