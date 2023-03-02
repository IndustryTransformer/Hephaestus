# using NNlib: Dropout
using Lux
includet("TransformersLite.jl")
using .TransformersLite


TransformerClassifier(
    Embed(32, 7455),                    # 238_560 parameters
    PositionEncoding(32),
    Dropout(0.1),
    TransformerEncoderBlock(
        MultiheadAttention(nhead = 4, dh = 16, dm = 8, dout = 32 => 32)(
            denseQ = Dense(32 => 32),         # 1_056 parameters
            denseK = Dense(32 => 32),         # 1_056 parameters
            denseV = Dense(32 => 32),         # 1_056 parameters
            denseO = Dense(32 => 32),         # 1_056 parameters
        ),
        Dropout(0.1),
        LayerNorm(32),                      # 64 parameters
        Dense(32 => 128, relu),             # 4_224 parameters
        Dense(128 => 32),                   # 4_128 parameters
        Dropout(0.1),
        LayerNorm(32),                      # 64 parameters
    ),
    Dense(32 => 1),                       # 33 parameters
    FlattenLayer(),
    Dense(50 => 5),                       # 255 parameters
)        # Total: 21 trainable arrays, 251_552 parameters,
# plus 1 non-trainable, 32_000 parameters, summarysize 1.083 MiB
