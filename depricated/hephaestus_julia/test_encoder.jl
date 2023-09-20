# %%
using Lux
using CUDA
using Transformers
using Transformers.Layers
using Transformers.TextEncoders

labels = map(string, 1:10)
startsym = "<s>"
endsym = "</s>"
unksym = "<unk>"
labels = [unksym, startsym, endsym, labels...]

textenc = TransformerTextEncoder(split, labels; startsym, endsym, unksym, padsym = unksym)

sample_data() = (d = join(map(string, rand(1:10, 10)), ' '); (d, d))

@show sample = sample_data()
# encode single sentence
@show encoded_sample_1 = encode(textenc, sample[1])
# encode for both encoder and decoder input
@show encoded_sample = encode(textenc, sample[1], sample[2])



N = 2
hidden_dim = 512
head_num = 8
head_dim = 64
ffn_dim = 2048

# define a Word embedding layer which turn word index to word vector
word_embed = Embed(hidden_dim, length(textenc.vocab)) |> todevice




pos_embed = SinCosPositionEmbed(hidden_dim)

# define 2 layer of transformer
encoder_trf =
    Transformer(TransformerBlock, N, head_num, hidden_dim, head_dim, ffn_dim) |> todevice

# define 2 layer of transformer decoder
decoder_trf =
    Transformer(TransformerDecoderBlock, N, head_num, hidden_dim, head_dim, ffn_dim) |>
    todevice

# define the layer to get the final output probabilities
# sharing weights with `word_embed`, don't/can't use `todevice`.
embed_decode = EmbedDecoder(word_embed)

function embedding(input)
    we = word_embed(input.token)
    pe = pos_embed(we)
    return we .+ pe
end

function encoder_forward(input)
    attention_mask = get(input, :attention_mask, nothing)
    e = embedding(input)
    t = encoder_trf(e, attention_mask) # return a NamedTuples (hidden_state = ..., ...)
    return t.hidden_state
end

function decoder_forward(input, m)
    attention_mask = get(input, :attention_mask, nothing)
    cross_attention_mask = get(input, :cross_attention_mask, nothing)
    e = embedding(input)
    t = decoder_trf(e, m, attention_mask, cross_attention_mask) # return a NamedTuple (hidden_state = ..., ...)
    p = embed_decode(t.hidden_state)
    return p
end




enc = encoder_forward(todevice(encoded_sample.encoder_input))
logits = decoder_forward(todevice(encoded_sample.decoder_input), enc)




using Lux.Losses # for logitcrossentropy

# define loss function
function shift_decode_loss(logits, trg, trg_mask)
    label = trg[:, 2:end, :]
    return logitcrossentropy(@view(logits[:, 1:end-1, :]), label, trg_mask - 1)
end

function loss(input)
    enc = encoder_forward(input.encoder_input)
    logits = decoder_forward(input.decoder_input, enc)
    ce_loss = shift_decode_loss(
        logits,
        input.decoder_input.token,
        input.decoder_input.attention_mask,
    )
    return ce_loss
end

# collect all the parameters
ps = Lux.params(word_embed, encoder_trf, decoder_trf)
opt = ADAM(1e-4)

# function for created batched data
using Transformers.Datasets: batched

# flux function for update parameters
using Lux: gradient
using Lux.Optimise: update!

preprocess(sample) = todevice(encode(textenc, sample[1], sample[2]))

# define training loop
function train!()
    @info "start training"
    for i = 1:2000
        sample = batched([sample_data() for i = 1:32]) # create 32 random sample and batched
        input = preprocess(sample)
        grad = gradient(() -> loss(input), ps)
        if i % 8 == 0
            l = loss(input)
            println("loss = $l")
        end
        update!(opt, ps, grad)
    end
end

train!()