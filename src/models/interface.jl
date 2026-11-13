"""
Společný nadtyp všech modelů. Každý model implementuje:

- `fit!(model, train::WindowedData; val = nothing)` – natrénuje model a vrátí ho
  (validační data využijí jen modely, které je potřebují, ostatní je ignorují),
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