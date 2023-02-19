using Lux, Random, NNlib, Zygote

struct LuxLinear <: Lux.AbstractExplicitLayer
    init_A::Any
    init_B::Any
end

function LuxLinear(A::AbstractArray, B::AbstractArray)
    # Storing Arrays or any mutable structure inside a Lux Layer is not recommended
    # instead we will convert this to a function to perform lazy initialization
    return LuxLinear(() -> copy(A), () -> copy(B))
end

# `B` is a parameter
Lux.initialparameters(rng::AbstractRNG, layer::LuxLinear) = (B = layer.init_B(),)

# `A` is a state
Lux.initialstates(rng::AbstractRNG, layer::LuxLinear) = (A = layer.init_A(),)

(l::LuxLinear)(x, ps, st) = st.A * ps.B * x, st
