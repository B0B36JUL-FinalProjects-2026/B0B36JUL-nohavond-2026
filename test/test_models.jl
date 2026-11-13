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
        @test length(model.history) == length(model.val_history) == 3
        @test predict(model, test) isa PointPrediction
    end

    lstm = fit!(LSTMModel(; head = GaussianHead(), hidden = 8, dense = 4, epochs = 3, batchsize = 16), train)
    p = predict(lstm, test)
    @test p isa GaussianPrediction
    @test length(p.μ) == length(test)
    @test all(>(0), p.σ)
    @test isempty(lstm.val_history)
end
