
## %%
using Gadfly, Cairo
using CSV
using DataFrames, DataFramesMeta
## %%

# df = CSV.read("/Users/kailukowiak/Library/CloudStorage/GoogleDrive-kai.lukowiak@industrytransformer.io/My Drive/Colab Notebooks/Hephaestus/data/HepheastusLoss2023-09-11T14_22_05.csv", DataFrame)
df = CSV.read("/Users/kailukowiak/Library/CloudStorage/GoogleDrive-kai.lukowiak@industrytransformer.io/My Drive/Colab Notebooks/Hephaestus/data/HepheastusLoss2023-09-11T14_30_50.csv", DataFrame)
# Remove `"Column1"` column
df = select(df, Not(:Column1))
# Rename "Hepheastus No Fine Tune" to "Hepheastus no Pretraining"
df = @transform(df, model = ifelse.(:model .== "Hephaestus No Fine Tune", "IndustryTransformer no Pretraining", :model))
df = @transform(df, model = ifelse.(:model .== "Hephaestus", "IndustryTransformer", :model))
first(df, 5)
# %%
plot(df, x=:n_rows, y=:test_loss, color=:model,
    Geom.line, Geom.point, Guide.xlabel("Amount of Labeled Data (Output)"),
    Guide.ylabel("Test loss"),
    Guide.title("Test loss for different models"))

# Scale x-axis on log scale
p = plot(df, x=:n_rows, y=:test_loss, color=:model,
    Geom.line, Geom.point, Guide.xlabel("Amount of Labeled Data"),
    Guide.ylabel("Validation Error"),
    Guide.title("Validation Error for Different Models"),
    Scale.x_log10,
    # Scale.y_log10
)

# Save plot
draw(PNG("./images/test_loss.png", 6inch, 4inch), p)
## %%

# Gadfly.push_theme(:dark)
# using dataset

p = plot(@subset(df, :model .!= "IndustryTransformer no Pretraining"), x=:n_rows, y=:test_loss, color=:model,
    Geom.line, Geom.point, Guide.xlabel("Amount Labelled Data (Output)"),
    Guide.ylabel("Test loss"),
    Guide.title("Test loss for different models"),
    Scale.x_log10,
    # Theme(default_color=color(["white", "red"])),
    # Guide.manual_color_key("Legend", ["Hephaestus", "XGBoost"], ["green", "deepskyblue"])
    # Scale.y_log10
)
p
# Save plot

# draw(PNG("test_loss_without_np.png", 6inch, 4inch), p)
## %%

# Creat a percent improvement column by spreading the data frame on the model column
df2 = unstack(df, :model, :test_loss)
# df2 = @transform(df, percent_improvement = (:test_loss .- :test_loss[1]) ./ :test_loss[1] .* 100)
first(df2, 4)
df2[!, "improvement"] = (df2[!, "XGBoost"] .- df2[!, "IndustryTransformer"]) ./ df2[!, "XGBoost"] .* 100
# %%
# Plot percent improvement
p_percent = plot(df2, x=:n_rows, y=:improvement,
    Geom.line, Geom.point, Guide.xlabel("Amount of Labeled Data (Output)"),
    Guide.ylabel("Percent improvement"),
    Guide.title("IndustryTransformer vs XGBoost (Accuracy Improvement)"),
    Scale.x_log10)

# Save plot
draw(PNG("./images/percent_improvement.png", 6inch, 4inch), p_percent)