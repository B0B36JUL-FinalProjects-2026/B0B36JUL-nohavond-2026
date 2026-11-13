function synthetic_windows(n; k = 4)
    X = randn(5, k, n)
    y = 5 .+ 0.3 .* vec(sum(X[5, :, :]; dims = 1)) .+ 0.1 .* randn(n)
    return WindowedData(X, y, fill(DateTime(2020), n), zeros(n), zeros(n))
end

@testset "TemporalAttention" begin
    a = TemporalAttention(8)
    h = randn(Float32, 8, 4, 3)
    @test size(a(h)) == (8, 3)
    w = EarthquakePredictionJapan.attention_weights(a, h)
    @test size(w) == (4, 3)
    @test all(sum(w; dims = 1) .≈ 1)
end

@testset "Network models" begin
    train, test = synthetic_windows(64), synthetic_windows(16)
    for model in (MLP(; hidden = [8], epochs = 3, batchsize = 16), LSTMModel(; hidden = 8, dense = 4, epochs = 3, batchsize = 16))
        @test model isa NetworkModel
        @test_throws ErrorException predict(model, test)
        fit!(model, train; val = test)
        @test length(model.state.history) == length(model.state.val_history) == 3
        @test predict(model, test) isa PointPrediction
    end

    mlp = fit!(MLP(; head = GaussianHead(), hidden = [8], epochs = 2, batchsize = 16), train)
    restored = load_state!(MLP(; head = GaussianHead(), hidden = [8]), model_state(mlp))
    @test predict(restored, test).μ == predict(mlp, test).μ
    @test restored.state.history == mlp.state.history

    lstm = fit!(LSTMModel(; head = GaussianHead(), hidden = 8, dense = 4, epochs = 3, batchsize = 16), train)
    p = predict(lstm, test)
    @test p isa GaussianPrediction
    @test length(p.μ) == length(test)
    @test all(>(0), p.σ)
    @test isempty(lstm.state.val_history)

    restored = load_state!(LSTMModel(; head = GaussianHead(), hidden = 8, dense = 4), model_state(lstm))
    @test predict(restored, test).σ == p.σ
end

@testset "Jednotné fit!" begin
    train, val = synthetic_windows(64), synthetic_windows(16)
    for model in (MeanBaseline(), RandomForest(; n_trees = 5), MLP(; hidden = [4], epochs = 1, batchsize = 16))
        @test fit!(model, train; val) === model
        @test length(predict(model, val)) == length(val)
    end
end
