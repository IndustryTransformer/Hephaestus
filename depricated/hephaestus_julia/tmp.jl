# %%
using Lux
using Random
# %%
d1 = Dense(10, 10)
d2 = Dense(10, 10)

# %%
rng = Random.default_rng()
Random.seed!(rng, 1234)

w_emb = rand(rng, 10)
w_flt1 = zeros(10)
w_flt2 = zeros(10)
w_flt3 = zeros(10)
w_flt1[1] = 1.0
w_flt2[1] = 2.0
w_flt3[1] = 3.0
# %%
ps1, st1 = Lux.setup(rng, d1)
ps2, st2 = Lux.setup(rng, d2)
d1(w_emb, ps1, st1)
d1(w_flt, ps1, st1)


d2(w_flt, ps2, st2)

d1(zeros(10), ps1, st1)

# %%
v1 = d1(w_flt1, ps1, st1)
v2 = d1(w_flt2, ps1, st1)
v3 = d1(w_flt3, ps1, st1)

# %%
v1[1] ./ v2[1]

v3[1] ./ v1[1]

w_flt22 = zeros(10)
w_flt21 = zeros(10)
w_flt22[2] = 2.0
w_flt21[2] = 1.0
v22 = d1(w_flt22, ps1, st1)
v21 = d1(w_flt21, ps1, st1)
v22[1] ./ v1[1]

v22[1] ./ v21[1]

both = d1(hcat(w_flt1, w_emb), ps1, st1)

both[1]
d1(w_flt1, ps1, st1)[1]
d1(w_emb, ps1, st1)[1]