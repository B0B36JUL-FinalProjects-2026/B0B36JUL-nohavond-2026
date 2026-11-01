const WINDOW_FEATURES = [:log_dt_days, :latitude, :longitude, :depth, :magnitude]

"""
    event_features(df) -> Matrix{Float64}

Příznaky každé události jako matice `5 × n`: `log1p` počtu dní od předchozí
události (u první 0), šířka, délka, hloubka a magnitude.
"""
function event_features(df::DataFrame)
    ms = Dates.value.(df.time)
    dt_days = [0.0; diff(ms) ./ 86_400_000]
    return permutedims(hcat(log1p.(dt_days), df.latitude, df.longitude, df.depth, df.magnitude))
end

"""
    WindowedData

Okna posledních `k` událostí a magnitudo události, která po nich následuje.

- `X`: příznaky ve tvaru `příznaky × k × počet oken` (tvar, který bere LSTM ve Flux.jl)
- `y`: magnitudo následující události
- `time`, `latitude`, `longitude`: čas a poloha následující události (pro mapy)
"""
struct WindowedData
    X::Array{Float64,3}
    y::Vector{Float64}
    time::Vector{DateTime}
    latitude::Vector{Float64}
    longitude::Vector{Float64}
end

Base.length(w::WindowedData) = length(w.y)

"""
    make_windows(df; k = 10) -> WindowedData

Z chronologicky seřazeného katalogu vytvoří klouzavá okna: z událostí
`i-k, …, i-1` se predikuje magnitude události `i`.
"""
function make_windows(df::DataFrame; k::Integer = 10)
    n = nrow(df)
    n > k || throw(ArgumentError("Katalog má $n událostí, na okno délky $k je potřeba víc než $k."))

    F = event_features(df)
    targets = k+1:n
    X = Array{Float64,3}(undef, size(F, 1), k, length(targets))
    for (j, i) in enumerate(targets)
        X[:, :, j] = F[:, i-k:i-1]
    end

    return WindowedData(X, df.magnitude[targets], df.time[targets], df.latitude[targets], df.longitude[targets])
end

"""
    flatten_windows(w::WindowedData) -> Matrix{Float64}

Zploští každé okno do jednoho sloupce (`5k × počet oken`) pro modely,
které neumí sekvence (Random Forest, MLP).
"""
flatten_windows(w::WindowedData) = reshape(w.X, :, length(w))