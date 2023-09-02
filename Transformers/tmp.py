epochs = 100
batch_size = 1000
lr = 0.5

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.8,
    patience=100,
    threshold=0.001,
    threshold_mode="rel",
    cooldown=0,
    min_lr=0.0001,
    eps=1e-08,
    verbose=False,
)


batch_count = 0
model_name = "tab_transformer"
model_run_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
model.train()
writer = SummaryWriter(f"runs/{model_name}_{model_run_time}")
for epoch in range(epochs):
    for i in range(0, X_train_num_tensor.size(0), batch_size):
        optimizer.zero_grad()
        y_pred = model(
            X_train_num_tensor[i : i + batch_size, :],
            X_train_cat_tensor[i : i + batch_size, :],
        )
        loss = loss_fn(y_pred, y_train_tensor[i : i + batch_size, :])
        loss.backward()
        optimizer.step()
        learning_rate = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/Regression_loss", loss, global_step=batch_count)
        writer.add_scalar(
            "Metrics/learning_rate",
            learning_rate,
            global_step=batch_count,
        )
        scheduler.step(loss)
        batch_count += 1

        if batch_count % 100 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} Loss: {loss.item():,.2f}, LR: {learning_rate}"
            )
