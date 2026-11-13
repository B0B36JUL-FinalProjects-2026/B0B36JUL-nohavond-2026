using CairoMakie

const LAST_N = 200

"""
    interval_axis!(pos, prediction, y; color)

Predikce s intervalem `INTERVAL` a skutečné hodnoty pro posledních `LAST_N`
událostí testu.
"""
function interval_axis!(pos, prediction, y; color)
    lower, upper = interval(prediction, INTERVAL)
    pct = round(Int, 100 * INTERVAL)
    ax = Axis(pos; title = "Predikce s $(pct)% intervalem (posledních $LAST_N událostí)",
        xlabel = "Pořadí události v testu", ylabel = "Magnitudo")
    idx = max(1, length(y) - LAST_N + 1):length(y)
    band!(ax, idx, lower[idx], upper[idx]; color = (color, 0.25), label = "$(pct)% interval")
    lines!(ax, idx, point(prediction)[idx]; color, label = "predikce")
    scatter!(ax, idx, y[idx]; color = :black, markersize = 5, label = "skutečnost")
    axislegend(ax; position = :lt)
    return ax
end

"""
    network_figure(model, test, file)

Průběh tréninku (NLL) a predikce s intervalem pro síť s `GaussianHead`.
"""
function network_figure(model::NetworkModel, test, file)
    fig = Figure(size = (1100, 450))
    ax = Axis(fig[1, 1]; title = "Průběh tréninku", xlabel = "Epocha", ylabel = "Loss")
    lines!(ax, model.state.history; label = "train (NLL)")
    lines!(ax, model.state.val_history; label = "val (NLL)")
    axislegend(ax)
    interval_axis!(fig[1, 2], predict(model, test), test.y; color = :darkorange)
    save_figure(fig, file)
end

function save_figure(fig, name)
    mkpath(FIGURES_DIR)
    save(joinpath(FIGURES_DIR, name), fig)
end
