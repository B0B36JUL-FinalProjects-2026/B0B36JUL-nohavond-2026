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
    chronological_split(df; train = 0.7, val = 0.15)

Rozdělí katalog seřazený podle času na tři navazující části: nejstarší události
jsou trénovací, pak validační a nejnovější testovací.
"""
function chronological_split(df::DataFrame; train::Real = 0.7, val::Real = 0.15)
    0 < train && 0 <= val && train + val < 1 ||
        throw(ArgumentError("Musí platit 0 < train, 0 ≤ val a train + val < 1."))
    issorted(df.time) || throw(ArgumentError("Katalog musí být seřazený podle času."))

    n = nrow(df)
    n_train = floor(Int, train * n)
    n_val = floor(Int, val * n)

    return DataSplit(
        df[1:n_train, :],
        df[n_train+1:n_train+n_val, :],
        df[n_train+n_val+1:end, :],
    )
end