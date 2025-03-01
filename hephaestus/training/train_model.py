import os
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from hephaestus.data.time_series_ds import TimeSeriesDS
from hephaestus.training.training import (
    compute_batch_loss,
    create_metric_history,
    eval_step,
    train_step,
)


def train_model(
    model: nn.Module,
    train_dataset: TimeSeriesDS,
    val_dataset: TimeSeriesDS,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    log_dir: str = "logs",
    save_dir: str = "models",
    device: torch.device = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    explosion_threshold: float = 10.0,
    max_explosions_per_epoch: int = 5,
    writer: Optional[SummaryWriter] = None,
) -> Dict:
    """Train a model with separate handling for numeric and categorical components.
    
    Args:
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Base learning rate for optimization
        log_dir: Directory for logs
        save_dir: Directory to save models
        device: Device to train on
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        explosion_threshold: Threshold to detect gradient explosions
        max_explosions_per_epoch: Maximum number of explosions before reducing LR
        writer: Optional TensorBoard writer
        
    Returns:
        Dictionary of training history
    """
    # Set default device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"TensorBoard log directory: {log_dir}")
    
    # Create TensorBoard writer if not provided
    if writer is None:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Adjust based on your system
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Create separate optimizers for numeric and categorical parts
    # Get all parameters
    all_params = list(model.parameters())
    
    # Filter categorical parameters - parameters from dense layers in categorical path
    categorical_params = []
    for name, param in model.named_parameters():
        if "categorical_dense" in name or "categorical_ln" in name:
            categorical_params.append(param)
    
    # Filter numeric parameters - all other parameters
    numeric_params = [p for p in all_params if p not in categorical_params]
    
    # Create optimizers with different learning rates
    numeric_optimizer = AdamW(numeric_params, lr=learning_rate)
    # Use a slightly higher learning rate for categorical to help it converge
    categorical_optimizer = AdamW(categorical_params, lr=learning_rate * 1.5)
    
    # Learning rate schedulers could be added here if needed
    
    # Initialize tracking variables
    history = create_metric_history()
    best_val_loss = float("inf")
    explosion_count = 0
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        
        # Training metrics
        train_loss = 0.0
        train_numeric_loss = 0.0
        train_categorical_loss = 0.0
        batch_count = 0
        
        # Progress bar
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        # Training step
        start_time = time.time()
        for i, batch in enumerate(train_pbar):
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(
                numeric_inputs=batch.numeric, 
                categorical_inputs=batch.categorical
            )
            
            # Compute losses
            total_loss, component_losses = compute_batch_loss(outputs, batch)
            
            # Scale loss for gradient accumulation
            scaled_loss = total_loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            # Update training metrics
            train_loss += total_loss.item()
            train_numeric_loss += component_losses.get("numeric_loss", 0.0)
            train_categorical_loss += component_losses.get("categorical_loss", 0.0)
            batch_count += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                "loss": f"{train_loss/batch_count:.4f}",
                "num_loss": f"{train_numeric_loss/batch_count:.4f}",
                "cat_loss": f"{train_categorical_loss/batch_count:.4f}"
            })
            
            # Perform optimizer step after accumulating gradients
            if (i + 1) % gradient_accumulation_steps == 0:
                # Check for exploding gradients
                numeric_grad_norm = torch.nn.utils.clip_grad_norm_(
                    numeric_params, max_grad_norm
                )
                categorical_grad_norm = torch.nn.utils.clip_grad_norm_(
                    categorical_params, max_grad_norm
                )
                
                # Detect gradient explosion
                if (numeric_grad_norm > explosion_threshold or 
                    categorical_grad_norm > explosion_threshold):
                    explosion_count += 1
                    if explosion_count > max_explosions_per_epoch:
                        print(f"Too many gradient explosions ({explosion_count}), reducing learning rate")
                        for param_group in numeric_optimizer.param_groups:
                            param_group["lr"] *= 0.5
                        for param_group in categorical_optimizer.param_groups:
                            param_group["lr"] *= 0.5
                        explosion_count = 0
                
                # Optimizer step
                numeric_optimizer.step()
                categorical_optimizer.step()
                numeric_optimizer.zero_grad()
                categorical_optimizer.zero_grad()
        
        # Calculate average training metrics
        avg_train_loss = train_loss / batch_count
        avg_train_numeric_loss = train_numeric_loss / batch_count
        avg_train_categorical_loss = train_categorical_loss / batch_count
        
        # Validation step
        model.eval()
        val_loss = 0.0
        val_numeric_loss = 0.0
        val_categorical_loss = 0.0
        val_batch_count = 0
        
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        
        with torch.no_grad():
            for batch in val_pbar:
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass
                outputs = model(
                    numeric_inputs=batch.numeric, 
                    categorical_inputs=batch.categorical
                )
                
                # Compute losses
                total_loss, component_losses = compute_batch_loss(outputs, batch)
                
                # Update validation metrics
                val_loss += total_loss.item()
                val_numeric_loss += component_losses.get("numeric_loss", 0.0)
                val_categorical_loss += component_losses.get("categorical_loss", 0.0)
                val_batch_count += 1
                
                # Update progress bar
                val_pbar.set_postfix({
                    "val_loss": f"{val_loss/val_batch_count:.4f}",
                    "val_num_loss": f"{val_numeric_loss/val_batch_count:.4f}",
                    "val_cat_loss": f"{val_categorical_loss/val_batch_count:.4f}"
                })
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / val_batch_count
        avg_val_numeric_loss = val_numeric_loss / val_batch_count
        avg_val_categorical_loss = val_categorical_loss / val_batch_count
        
        # Log to TensorBoard
        writer.add_scalar("Loss/train/total", avg_train_loss, epoch)
        writer.add_scalar("Loss/train/numeric", avg_train_numeric_loss, epoch)
        writer.add_scalar("Loss/train/categorical", avg_train_categorical_loss, epoch)
        writer.add_scalar("Loss/val/total", avg_val_loss, epoch)
        writer.add_scalar("Loss/val/numeric", avg_val_numeric_loss, epoch)
        writer.add_scalar("Loss/val/categorical", avg_val_categorical_loss, epoch)
        
        # Log learning rates
        writer.add_scalar("LearningRate/numeric", numeric_optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("LearningRate/categorical", categorical_optimizer.param_groups[0]["lr"], epoch)
        
        # Update history
        history["train_loss"].append(avg_train_loss)
        history["train_numeric_loss"].append(avg_train_numeric_loss)
        history["train_categorical_loss"].append(avg_train_categorical_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_numeric_loss"].append(avg_val_numeric_loss)
        history["val_categorical_loss"].append(avg_val_categorical_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "numeric_optimizer_state_dict": numeric_optimizer.state_dict(),
                    "categorical_optimizer_state_dict": categorical_optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                os.path.join(save_dir, "best_model.pt")
            )
            print(f"Saved best model with validation loss: {avg_val_loss:.4f}")
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train Numeric Loss: {avg_train_numeric_loss:.4f}, Val Numeric Loss: {avg_val_numeric_loss:.4f}")
        print(f"Train Cat Loss: {avg_train_categorical_loss:.4f}, Val Cat Loss: {avg_val_categorical_loss:.4f}")
        
        # Print current learning rates
        print(f"Current numeric learning rate: {numeric_optimizer.param_groups[0]['lr']:.2e}")
        print(f"Current categorical learning rate: {categorical_optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 50)
    
    # Training complete
    print(f"Training completed! TensorBoard logs saved to {log_dir}")
    print("To view training metrics, run:")
    print(f"tensorboard --logdir={log_dir}")
    
    return history
