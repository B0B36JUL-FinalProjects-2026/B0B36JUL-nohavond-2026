"""
    TemporalAttention(hidden; init = Flux.glorot_uniform)

Pozornost přes časové kroky. Pro každý skrytý stav LSTM spočítá skóre,
`softmax` přes čas z nich udělá váhy a výstupem je vážený součet stavů.

Vstup `hidden × délka × batch`, výstup `hidden × batch`.
"""
struct TemporalAttention{D}
    score::D
end

Flux.@layer TemporalAttention

TemporalAttention(hidden::Integer; init = Flux.glorot_uniform) =
    TemporalAttention(Flux.Dense(hidden => 1; init))

"""
    attention_weights(a::TemporalAttention, h) -> Matrix

Váhy časových kroků ve tvaru `délka × batch` (každý sloupec má součet 1).
"""
attention_weights(a::TemporalAttention, h::AbstractArray{<:Real,3}) =
    dropdims(Flux.softmax(a.score(h); dims = 2); dims = 1)

function (a::TemporalAttention)(h::AbstractArray{<:Real,3})
    w = attention_weights(a, h)
    return dropdims(sum(h .* reshape(w, 1, size(w)...); dims = 2); dims = 2)
end
