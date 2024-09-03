# %%
using Lux
using NNlib
using Missings
using DataFrames, DataFramesMeta
using Parquet2
# %%

weather = Parquet2.Dataset("hephaestus_julia/ab_weather.parquet")
weather = DataFrame(weather, copycols = false)

# %%
# Get columns that aren't all null
weather = select(weather, Not([:id, :local_date]))
weather = weather[:, Not(all.(ismissing, eachcol(weather)))]
weather[!, r"local|flag"] = weather[!, r"local|flag"] .|> string
weather[!, names(weather, Union{String, Missing})] =
	string.(weather[!, names(weather, Union{String, Missing})])
# Replace missing string values with "missing"
# Scale the numeric columns

column_names = names(weather)
unique_string_values = Dict(
	col => unique(weather[!, col]) for
	col in column_names if nonmissingtype(eltype(weather[!, col])) <: String
)


text_tokens = String[]
for (k, v) in unique_string_values
	for val in v
		push!(text_tokens, val)
	end
end
text_tokens = vcat(text_tokens, column_names) |> unique |> sort

# %%"""
struct MultiheadAttention{Q <: Dense, K <: Dense, V <: Dense, O <: Dense}
	init_nhead::Int
	init_denseQ::Q
	init_denseK::K
	init_denseV::V
	init_denseO::O
end

Lux.initialparameters(rng::AbstractRNG, mha::MultiheadAttention) = (
	DenseQ = mha.init_denseQ,
	DenseK = mha.init_denseK,
	DenseV = mha.init_denseV,
	DenseO = mha.init_denseO,
)

Lux.initialstates(rng::AbstractRNG, mha::MultiheadAttention) = (nhead = mha.init_nhead)


"""
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

function (mha::MultiheadAttention, ps, st)(
	query::A1,
	key::A2,
	value::A3,
) where {
	T,
	A1 <: AbstractArray{T, 3},
	A2 <: AbstractArray{T, 3},
	A3 <: AbstractArray{T, 3},
}
	qs = size(query)
	ks = size(key)
	vs = size(value)

	Q = mha.DenseQ(query)
	K = mha.DenseK(key)
	V = mha.DenseV(value)

	dm = size(Q, 1)
	dh = div(dm, mha.nhead)
	#size(Q) == (dh*nhead, N, B) => (dh, nhead, N, B) => (dh, N, nhead, B)
	Q = permutedims(reshape(Q, dh, mha.nhead, qs[2], qs[3]), [1, 3, 2, 4])
	K = permutedims(reshape(K, dh, mha.nhead, ks[2], ks[3]), [1, 3, 2, 4])
	V = permutedims(reshape(V, dh, mha.nhead, vs[2], vs[3]), [1, 3, 2, 4])
	#size(A) == (dh, N, nhead, B)
	A = scaled_dot_attention(Q, K, V)
	#size(A) == (dh, N, nhead, B) => (dh, nhead, N, B) => (dm, N, B)
	A = permutedims(A, [1, 3, 2, 4])
	A = reshape(A, dm, size(A, 3), size(A, 4))

	mha.denseO(A), ps, st
end


Lux.initialparameters(rng::AbstractRNG, mha::MultiheadAttention) = (
	denseQ = initialparameters(rng, mha.denseQ),
	denseK = initialparameters(rng, mha.denseK),
	denseV = initialparameters(rng, mha.denseV),
	denseO = initialparameters(rng, mha.denseO),
)

struct PositionalEncoding{W <: Matrix{Float32}}
	encoding::W
end

function PositionalEncoding(d_model::Int, max_len::Int, n::Int = 10000)
	encoding = zeros(Float32, d_model, max_len)
	for pos ∈ 1:max_len
		for row ∈ 1:2:div(d_model, 2)
			denom = n^(-row / d_model)
			@inbounds encoding[row, pos] = sin(pos * denom)
			@inbounds encoding[row+1, pos] = cos(pos * denom)
		end
	end
	PositionalEncoding(encoding)
end



function Lux.initialparameters(rng::AbstractRNG, pe::PositionalEncoding)
	return encoding = pe.encoding
end


function NNLib.batched_mul(A::AbstractArray{T, 4}, B::AbstractArray{T, 4}) where {T}
	if (
		(size(A, 2) != size(B, 1)) ||
		(size(A, 3) != size(B, 3)) ||
		(size(A, 4) != size(B, 4))
	)
		message = "A has dimensions $(size(A)) but B has dimensions $(size(B))"
		throw(DimensionMismatch(message))
	end
	new_A = reshape(A, size(A, 1), size(A, 2), :)
	new_B = reshape(B, size(B, 1), size(B, 2), :)
	C = batched_mul(new_A, new_B)
	new_C = reshape(C, (size(C, 1), size(C, 2), size(A, 3), size(A, 4)))
	new_C
end

# %%

function mul4d(A::AbstractArray{T, 4}, B::AbstractArray{T, 4}) where {T}
	C = Array{Float64, 4}(undef, size(A, 1), size(B, 2), size(A, 3), size(A, 4))
	for l in axes(A, 4)
		for k in axes(A, 3)
			C[:, :, k, l] = A[:, :, k, l] * B[:, :, k, l]
		end
	end
	C
end


function mul4d_no_types(A::Matrix, B::Matrix)
	C = Array{Float64, 4}(undef, size(A, 1), size(B, 2), size(A, 3), size(A, 4))
	for l in axes(A, 4)
		for k in axes(A, 3)
			@inbounds C[:, :, k, l] = A[:, :, k, l] * B[:, :, k, l]
		end
	end
	C
end
# %%

println("Types...")
@code_warntype mul4d(rand(2, 2, 2, 2), rand(2, 2, 2, 2))
println("No types...")
@code_warntype mul4d_no_types(rand(2, 2, 2, 2), rand(2, 2, 2, 2))

using BenchmarkTools


@benchmark mul4d(rand(20, 20, 20, 20), rand(20, 20, 20, 20))
@benchmark mul4d_no_types(rand(20, 20, 20, 20), rand(20, 20, 20, 20))