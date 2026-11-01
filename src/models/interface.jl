"""
Společný nadtyp všech modelů. Každý model implementuje:

- `fit!(model, train::WindowedData)` – natrénuje model a vrátí ho,
- `predict(model, data::WindowedData)` – vrátí podtyp [`AbstractPrediction`](@ref).
"""
abstract type AbstractQuakeModel end

"""
Výstupní hlava modelu určuje, jaký typ predikce model vrací.
"""
abstract type AbstractHead end

struct PointHead <: AbstractHead end
struct SampleHead <: AbstractHead end
struct GaussianHead <: AbstractHead end


function fit! end
function predict end