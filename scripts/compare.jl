# Souhrn všech modelů z uložených predikcí v results/*.jld2.
# Nejdřív je potřeba spustit trénovací skripty train_*.jl.

include("common.jl")
include("plots.jl")

const RUNS = ["baseline", "random_forest", "mlp", "lstm"]
const LEVELS = 0.1:0.1:0.9
# Kategorické barvy v pevném pořadí (pravděpodobnostní modely v pořadí RUNS).
const SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

function load_runs(runs)
    names = String[]
    val_predictions, test_predictions = Pair{String,AbstractPrediction}[], Pair{String,AbstractPrediction}[]
    val_y = test_y = nothing
    for run in runs
        file = joinpath(RESULTS_DIR, "$run.jld2")
        if !isfile(file)
            @warn "Chybí $file, přeskakuji (spusť scripts/train_$run.jl)."
            continue
        end
        r = load(file)
        # Všechny běhy musí být vyhodnocené na stejných datech.
        if val_y === nothing
            val_y, test_y = r["val_y"], r["test_y"]
        else
            r["val_y"] == val_y && r["test_y"] == test_y ||
                error("$file má jiná val/test data než ostatní běhy, přetrénuj ho.")
        end
        for name in r["names"]
            push!(names, name)
            push!(val_predictions, name => r["val_predictions"][name])
            push!(test_predictions, name => r["test_predictions"][name])
        end
    end
    isempty(names) && error("Nenašel jsem žádné výsledky v $RESULTS_DIR.")
    return val_predictions, test_predictions, val_y, test_y
end

val_predictions, test_predictions, val_y, test_y = load_runs(RUNS)

report(io) = begin
    results_table(val_predictions, val_y, "Validační data"; io)
    results_table(test_predictions, test_y, "Testovací data"; io)
end
report(stdout)
open(report, joinpath(RESULTS_DIR, "summary.txt"), "w")

fig = Figure(size = (1100, 480))

# CRPS na testu: u bodových modelů je rovno MAE, takže jde srovnat všechny.
names = first.(test_predictions)
crps = [evaluate(CRPS(), p, test_y) for (_, p) in test_predictions]
ax1 = Axis(fig[1, 1]; title = "CRPS na testovacích datech (nižší je lepší)", xlabel = "CRPS",
    yticks = (eachindex(names), names), yreversed = true,
    xgridvisible = false, ygridvisible = false)
barplot!(ax1, eachindex(names), crps; direction = :x, color = SERIES_COLORS[1], gap = 0.3)
text!(ax1, crps, eachindex(names); text = fmt.(crps), align = (:left, :center),
    offset = (4, 0), fontsize = 11, color = :gray25)
xlims!(ax1, 0, 1.2 * maximum(crps))

# Kalibrace: skutečné pokrytí centrálních intervalů proti nominální úrovni.
ax2 = Axis(fig[1, 2]; title = "Kalibrace intervalů na testovacích datech",
    xlabel = "Nominální pokrytí", ylabel = "Skutečné pokrytí", aspect = 1)
lines!(ax2, [0, 1], [0, 1]; color = :gray60, linestyle = :dash, label = "ideální kalibrace")
probabilistic = filter(((_, p),) -> p isa ProbabilisticPrediction, test_predictions)
for (i, (name, p)) in enumerate(probabilistic)
    coverage = [evaluate(Coverage(l), p, test_y) for l in LEVELS]
    scatterlines!(ax2, collect(LEVELS), coverage; color = SERIES_COLORS[i], linewidth = 2,
        markersize = 8, label = name)
end
limits!(ax2, 0, 1, 0, 1)
axislegend(ax2; position = :lt, framevisible = false)

save_figure(fig, "summary.png")
