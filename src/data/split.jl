"""
    DataSplit(train, val, test)

Trénovací, validační a testovací část dat.
"""
struct DataSplit{T}
    train::T
    val::T
    test::T
end

"""
    chronological_split(data; train = 0.7, val = 0.15)

Rozdělí data seřazená podle času na tři navazující části: nejstarší jsou
trénovací, pak validační a nejnovější testovací. Funguje pro katalog
(`DataFrame`) i pro okna ([`WindowedData`](@ref)).

Okna je lepší dělit až po [`make_windows`](@ref) nad celým katalogem: začátek
validační a testovací části pak nepřijde o prvních `k` cílů a vstupní okna
na hranici můžou obsahovat události z předchozí části (ty jsou v době
predikce známé, takže nejde o únik informace).
"""
function chronological_split(data; train::Real = 0.7, val::Real = 0.15)
    0 < train && 0 <= val && train + val < 1 ||
        throw(ArgumentError("Musí platit 0 < train, 0 ≤ val a train + val < 1."))
    issorted(data.time) || throw(ArgumentError("Data musí být seřazená podle času."))

    n = nobs(data)
    n_train = floor(Int, train * n)
    n_val = floor(Int, val * n)

    return DataSplit(
        subset_rows(data, 1:n_train),
        subset_rows(data, n_train+1:n_train+n_val),
        subset_rows(data, n_train+n_val+1:n),
    )
end

nobs(df::DataFrame) = nrow(df)
nobs(w::WindowedData) = length(w)

subset_rows(df::DataFrame, idx) = df[idx, :]
subset_rows(w::WindowedData, idx) =
    WindowedData(w.X[:, :, idx], w.y[idx], w.time[idx], w.latitude[idx], w.longitude[idx])
