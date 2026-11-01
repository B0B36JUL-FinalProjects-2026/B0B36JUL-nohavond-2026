"""
    train_network!(network, head, train, val; epochs, batchsize, learning_rate, rng)

Společná trénovací smyčka pro neuronové sítě.
"""
function train_network!(network, head::AbstractHead, train, val;
    epochs::Integer, batchsize::Integer, learning_rate::Real, rng::AbstractRNG)

    X, y = train
    opt_state = Flux.setup(Flux.Adam(learning_rate), network)
    loader = Flux.DataLoader((X, y); batchsize, shuffle = true, rng)

    history = Float64[]
    val_history = Float64[]
    best_loss = Inf
    best_state = deepcopy(Flux.state(network))

    for _ in 1:epochs
        for (xb, yb) in loader
            grads = Flux.gradient(m -> loss(head, m(xb), yb), network)
            Flux.update!(opt_state, network, grads[1])
        end

        push!(history, loss(head, network(X), y))
        if val !== nothing
            current = loss(head, network(val[1]), val[2])
            push!(val_history, current)
            if current < best_loss
                best_loss = current
                best_state = deepcopy(Flux.state(network))
            end
        end
    end

    val === nothing || Flux.loadmodel!(network, best_state)
    return history, val_history
end