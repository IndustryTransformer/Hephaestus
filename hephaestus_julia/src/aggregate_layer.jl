"""
    MeanLayer(x)

Reduce to the mean along dims=1.

Compare Lux.GlobalMeanPool()
"""
struct MeanLayer end

Lux.@functor MeanLayer

function (m::MeanLayer)(x::AbstractArray)
    mean(x, dims = 1)
end

function Base.show(io::IO, m::MeanLayer)
    print(io, "MeanLayer()")
end

"""
FlattenLayer(x)

Return a matrix of nlayers × nbatch.
"""
struct FlattenLayer end

Lux.@functor FlattenLayer

function (f::FlattenLayer)(x::AbstractArray{T,3}) where {T}
    reshape(x, :, size(x, 3)) # same as Lux.flatten
end

function (f::FlattenLayer)(x::AbstractArray{T,2}) where {T}
    reshape(x, :, 1) # returns a column vector
end

function Base.show(io::IO, f::FlattenLayer)
    print(io, "FlattenLayer()")
end
