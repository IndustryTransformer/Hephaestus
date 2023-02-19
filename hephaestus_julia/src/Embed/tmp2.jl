using Flux, Random, NNlib, Zygote, Optimisers

struct FluxLinear
    A::Any
    B::Any
end







# `A` is not trainable
Optimisers.trainable(f::FluxLinear) = (B = f.B,)

# Needed so that both `A` and `B` can be transfered between devices
Flux.@functor FluxLinear

(l::FluxLinear)(x) = l.A * l.B * x