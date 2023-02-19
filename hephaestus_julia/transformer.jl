# %%
using Lux
using CUDA


struct MultiheadAttention
    nhead::Int
    densQ::Q
    densK::K
    densV::V
    densO::O
end

Lux.@functor MultiheadAttention (densQ, densK, densV, densO)

"""
    MultiheadAttention(nhead::Int, dm::Int, dh::Int, dout::Int)
    MultiheadAttention(nhead::Int, dm::Int, dout::Int)
Multihead dot product attention Layer. `nhead` is the number of heads, 
`dm` is the model embedding dimension size, `dh` is the size of each head, 
`dout` is the output size.
"""
function MultiheadAttention(nhead::Int, dm::Int, dh::Int, dout::Int)
    MultiheadAttention(
        nhead,
        Dense(dm, dh * nhead),
        Dense(dm, dh * nhead),
        Dense(dm, dh * nhead),
        Dense(dh * nhead, dout),
    )
end
