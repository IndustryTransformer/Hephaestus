
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
df = @transform(df, model = ifelse.(:model .== "Hephaestus No Fine Tune", "Hepheastus no Pretraining", :model))
first(df, 5)
# %%
plot(df, x=:n_rows, y=:test_loss, color=:model,
    Geom.line, Geom.point, Guide.xlabel("Number of rows"),
    Guide.ylabel("Test loss"),
    Guide.title("Test loss for different models"))

# Scale x-axis on log scale
p = plot(df, x=:n_rows, y=:test_loss, color=:model,
    Geom.line, Geom.point, Guide.xlabel("Number of rows"),
    Guide.ylabel("Test loss"),
    Guide.title("Test loss for different models"),
    Scale.x_log10,
    # Scale.y_log10
)

# Save plot
draw(PNG("test_loss.png", 6inch, 4inch), p)
## %%

# Creat a percent improvement column by spreading the data frame on the model column
df2 = unstack(df, :model, :test_loss)
# df2 = @transform(df, percent_improvement = (:test_loss .- :test_loss[1]) ./ :test_loss[1] .* 100)
first(df2, 4)
df2[!, "improvement"] = (df2[!, "XGBoost"] .- df2[!, "Hephaestus"]) ./ df2[!, "XGBoost"] .* 100
# %%
# Plot percent improvement
p_percent = plot(df2, x=:n_rows, y=:improvement,
    Geom.line, Geom.point, Guide.xlabel("Number of rows"),
    Guide.ylabel("Percent improvement"),
    Guide.title("Percent improvement for Pre-Trained Model"),
    Scale.x_log10)

# Save plot
draw(PNG("percent_improvement.png", 6inch, 4inch), p_percent)