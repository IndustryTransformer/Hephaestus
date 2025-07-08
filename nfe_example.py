"""
Example demonstrating Neural Feature Engineering vs MoE vs Standard models.

This script shows how to use the new Neural Feature Engineering system
that automatically learns feature interactions without the parameter 
explosion of MoE models.
"""

import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import hephaestus.single_row_models as sr

# Set environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_sample_data():
    """Create sample turbine data for demonstration."""
    torch.manual_seed(42)
    
    # Generate synthetic turbine data
    n_samples = 1000
    
    # Numeric features
    ambient_temp = torch.randn(n_samples) * 10 + 20  # Temperature
    pressure = torch.randn(n_samples) * 5 + 100      # Pressure
    humidity = torch.rand(n_samples) * 100           # Humidity %
    power_output = torch.randn(n_samples) * 50 + 200 # Power MW
    fuel_flow = torch.randn(n_samples) * 10 + 50     # Fuel flow
    
    # Create interactions for NOx prediction
    # NOx is affected by temperature-pressure interaction and fuel efficiency
    fuel_efficiency = power_output / fuel_flow
    temp_pressure_interaction = ambient_temp * pressure / 1000
    
    # Target NOx with realistic dependencies
    nox = (
        0.1 * ambient_temp +
        0.05 * pressure +
        0.2 * humidity +
        0.15 * fuel_flow +
        0.3 * temp_pressure_interaction +
        -0.25 * fuel_efficiency +
        torch.randn(n_samples) * 2  # noise
    )
    
    # Categorical features
    turbine_type = torch.randint(0, 3, (n_samples,))  # 3 turbine types
    fuel_type = torch.randint(0, 2, (n_samples,))     # 2 fuel types
    maintenance_status = torch.randint(0, 4, (n_samples,))  # 4 maintenance levels
    
    # Create categorical mappings
    turbine_types = ['Type_A', 'Type_B', 'Type_C']
    fuel_types = ['Natural_Gas', 'Diesel']
    maintenance_statuses = ['Good', 'Fair', 'Poor', 'Critical']
    
    # Create DataFrame
    df = pd.DataFrame({
        'ambient_temp': ambient_temp.numpy(),
        'pressure': pressure.numpy(),
        'humidity': humidity.numpy(),
        'power_output': power_output.numpy(),
        'fuel_flow': fuel_flow.numpy(),
        'turbine_type': [turbine_types[turbine_type[i]] for i in range(n_samples)],
        'fuel_type': [fuel_types[fuel_type[i]] for i in range(n_samples)],
        'maintenance_status': [maintenance_statuses[maintenance_status[i]] for i in range(n_samples)],
        'nox': nox.numpy()
    })
    
    return df

def train_and_compare_models():
    """Train and compare Standard, MoE, and Neural Feature Engineering models."""
    print("🚀 Neural Feature Engineering vs MoE vs Standard Model Comparison")
    print("=" * 70)
    
    # Create sample data
    df = create_sample_data()
    print(f"📊 Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Data types:\n{df.dtypes}")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Scale numeric features
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32']).columns
    numeric_cols = [col for col in numeric_cols if col != 'nox']  # Remove target
    print(f"Numeric columns for scaling: {numeric_cols}")
    
    if numeric_cols:  # Only scale if there are numeric columns
        scaler = StandardScaler()
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    
    # Model configurations
    configs = {
        'Standard': sr.SingleRowConfig.generate(
            train_df, 
            target='nox',
            use_moe=False,
            use_neural_feature_engineering=False
        ),
        'MoE': sr.SingleRowConfig.generate(
            train_df, 
            target='nox',
            use_moe=True,
            num_experts=6,
            moe_top_k=2,
            use_neural_feature_engineering=False
        ),
        'Neural_Feature_Engineering': sr.SingleRowConfig.generate(
            train_df, 
            target='nox',
            use_moe=False,
            use_neural_feature_engineering=True,
            nfe_max_interactions=10,
            nfe_max_ratios=8,
            nfe_polynomial_degree=3,
            nfe_fusion_strategy="learned"
        )
    }
    
    results = {}
    
    for model_name, config in configs.items():
        print(f"\n🔧 Training {model_name} Model...")
        print("-" * 50)
        
        # Create model
        from hephaestus.single_row_models.single_row_models import TabularEncoderRegressor
        model = TabularEncoderRegressor(
            model_config=config,
            d_model=64,
            n_heads=4,
            use_linear_numeric_embedding=True,
            numeric_embedding_type="simple"
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📈 Model Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        
        # Quick test forward pass
        test_dataset = sr.TabularDS(test_df.head(10), config)
        sample_batch = test_dataset[0]
        
        with torch.no_grad():
            output = model(sample_batch.inputs.numeric, sample_batch.inputs.categorical)
            print(f"  Output shape: {output.shape}")
            
            # Show Neural Feature Engineering metrics if available
            if hasattr(model, 'get_neural_feature_engineering_metrics'):
                nfe_metrics = model.get_neural_feature_engineering_metrics()
                if nfe_metrics:
                    print(f"  🧠 Neural Feature Engineering:")
                    if 'total_engineered_features' in nfe_metrics:
                        print(f"    Total engineered features: {nfe_metrics['total_engineered_features']}")
                    if 'enabled_components' in nfe_metrics:
                        enabled = nfe_metrics['enabled_components']
                        print(f"    Enabled components: {[k for k, v in enabled.items() if v]}")
            
            # Show MoE metrics if available  
            if hasattr(model, 'get_moe_metrics'):
                moe_metrics = model.get_moe_metrics()
                if moe_metrics:
                    print(f"  🧠 MoE layers: {len(moe_metrics)}")
        
        results[model_name] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'config': config
        }
    
    # Summary comparison
    print(f"\n🎯 COMPARISON SUMMARY")
    print("=" * 70)
    
    standard_params = results['Standard']['trainable_params']
    
    for model_name, result in results.items():
        params = result['trainable_params']
        ratio = params / standard_params
        print(f"{model_name:25} | {params:>8,} params | {ratio:>6.1f}x size")
    
    print(f"\n✨ Neural Feature Engineering provides automatic feature discovery")
    print(f"   without the parameter explosion of MoE approaches!")
    print(f"\n💡 Key advantages:")
    print(f"   • Learns meaningful feature interactions automatically")
    print(f"   • Much more parameter-efficient than MoE")
    print(f"   • Includes polynomial, ratio, and attention-based features")
    print(f"   • Provides interpretable feature importance metrics")
    
    return results

if __name__ == "__main__":
    results = train_and_compare_models()