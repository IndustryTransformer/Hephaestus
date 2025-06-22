#!/usr/bin/env python3
"""
Big Synthetic Dataset Generator for Next Row Generation Testing

Creates sequences of 128 rows with the same idx, featuring:
- Sin/cos waves with different amplitudes and periods per group
- FizzBuzz with random start
- Random integers and lagged multiplication
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


class BigSyntheticDataset:
    """
    Generate large synthetic dataset with predictable patterns for next row generation.

    Features:
    - Groups of 128 rows with same idx
    - Sin/cos waves with varying amplitudes (1-5) and periods per group
    - FizzBuzz with random start (0-100)
    - Random integer column (0-100)
    - Lagged multiplication column (5 rows back * 2.5, NaN for first 5)
    """

    def __init__(
        self,
        n_groups: int = 10000,
        group_size: int = 128,
        random_seed: int | None = None,
    ):
        self.n_groups = n_groups
        self.group_size = group_size
        self.total_rows = n_groups * group_size

        if random_seed is not None:
            np.random.seed(random_seed)

        print("🔧 Initializing dataset generator:")
        print(f"  Groups: {n_groups:,}")
        print(f"  Rows per group: {group_size}")
        print(f"  Total rows: {self.total_rows:,}")

    def _generate_fizzbuzz_sequence(self, start: int, length: int) -> list[str]:
        """Generate FizzBuzz sequence starting from a given number."""
        result = []
        for i in range(start, start + length):
            if i % 15 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                result.append("Fizz")
            elif i % 5 == 0:
                result.append("Buzz")
            else:
                result.append("NUMBER")  # Simplified: all numbers become "NUMBER"
        return result

    def _generate_group_data(self, group_id: int) -> dict:
        """Generate data for a single group of 128 rows."""
        # Group-specific parameters
        sin_amplitude = np.random.uniform(1, 5)
        cos_amplitude = np.random.uniform(1, 5)
        sin_period = np.random.uniform(8, 32)  # Period in rows
        cos_period = np.random.uniform(8, 32)
        fizzbuzz_start = np.random.randint(0, 101)

        # Row indices within group (0 to group_size-1)
        row_indices = np.arange(self.group_size)

        # Sin/cos waves
        sin_values = sin_amplitude * np.sin(2 * np.pi * row_indices / sin_period)
        cos_values = cos_amplitude * np.cos(2 * np.pi * row_indices / cos_period)

        # FizzBuzz sequence
        fizzbuzz_values = self._generate_fizzbuzz_sequence(
            fizzbuzz_start, self.group_size
        )

        # Random integers (0-100)
        random_ints = np.random.randint(0, 101, self.group_size)

        # Lagged multiplication (5 rows back * 2.5)
        lagged_mult = np.full(self.group_size, np.nan)
        for i in range(5, self.group_size):
            lagged_mult[i] = random_ints[i - 5] * 2.5

        return {
            "idx": [group_id] * self.group_size,
            "row_in_group": row_indices.tolist(),
            "sin_wave": sin_values.tolist(),
            "cos_wave": cos_values.tolist(),
            "fizzbuzz": fizzbuzz_values,
            "random_int": random_ints.tolist(),
            "lagged_mult": lagged_mult.tolist(),
            # Store group parameters for analysis
            "sin_amplitude": [sin_amplitude] * self.group_size,
            "cos_amplitude": [cos_amplitude] * self.group_size,
            "sin_period": [sin_period] * self.group_size,
            "cos_period": [cos_period] * self.group_size,
            "fizzbuzz_start": [fizzbuzz_start] * self.group_size,
        }

    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete dataset."""
        print("🔧 Generating dataset...")

        all_data = {
            "idx": [],
            "row_in_group": [],
            "sin_wave": [],
            "cos_wave": [],
            "fizzbuzz": [],
            "random_int": [],
            "lagged_mult": [],
            "sin_amplitude": [],
            "cos_amplitude": [],
            "sin_period": [],
            "cos_period": [],
            "fizzbuzz_start": [],
        }

        # Generate data group by group to manage memory
        for group_id in range(self.n_groups):
            if group_id % 1000 == 0:
                print(f"  Generated {group_id:,}/{self.n_groups:,} groups...")

            group_data = self._generate_group_data(group_id)

            for key in all_data:
                all_data[key].extend(group_data[key])

        df = pd.DataFrame(all_data)

        print("✅ Dataset generated successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        return df

    def save_dataset(self, df: pd.DataFrame, filename: str | None = None) -> Path:
        """Save dataset to parquet file."""
        if filename is None:
            script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            data_dir = script_dir / "data"
            data_dir.mkdir(exist_ok=True)
            filename = data_dir / f"big_synthetic_dataset_{self.n_groups}groups.parquet"
        else:
            filename = Path(filename)

        print(f"💾 Saving dataset to {filename}...")
        df.to_parquet(filename, compression="snappy")

        file_size = filename.stat().st_size / 1024**2
        print(f"✅ Dataset saved! File size: {file_size:.1f} MB")

        return filename

    def analyze_dataset(self, df: pd.DataFrame) -> None:
        """Print dataset analysis."""
        print("\n📊 Dataset Analysis:")
        print("=" * 50)

        print(f"Total rows: {len(df):,}")
        print(f"Unique groups (idx): {df['idx'].nunique():,}")
        print(f"Rows per group: {df.groupby('idx').size().iloc[0]}")

        print("\nColumn types:")
        for col in df.columns:
            dtype = df[col].dtype
            if col == "lagged_mult":
                null_count = df[col].isnull().sum()
                print(f"  {col}: {dtype} ({null_count:,} NaN values)")
            else:
                print(f"  {col}: {dtype}")

        print("\nSample of first group (idx=0):")
        sample = df[df["idx"] == 0].head(10)
        for col in [
            "row_in_group",
            "sin_wave",
            "cos_wave",
            "fizzbuzz",
            "random_int",
            "lagged_mult",
        ]:
            print(f"  {col}: {sample[col].tolist()}")

        print("\nParameter ranges across groups:")
        param_cols = [
            "sin_amplitude",
            "cos_amplitude",
            "sin_period",
            "cos_period",
            "fizzbuzz_start",
        ]
        group_params = df.groupby("idx")[param_cols].first()
        for col in param_cols:
            values = group_params[col]
            print(f"  {col}: {values.min():.2f} to {values.max():.2f}")


def main():
    """Generate and save the big synthetic dataset."""
    print("🧪 Big Synthetic Dataset Generator")
    print("=" * 50)

    # Create generator
    generator = BigSyntheticDataset(
        n_groups=10000,
        group_size=128,
        random_seed=42,
    )

    # Generate dataset
    df = generator.generate_dataset()

    # Analyze it
    generator.analyze_dataset(df)

    # Save it
    filepath = generator.save_dataset(df)

    print("\n✅ Complete! Dataset saved to:")
    print(f"  {filepath}")
    print("\nTo load in other scripts:")
    print(f"  df = pd.read_parquet('{filepath}')")
    print(f"  df_pl = pl.read_parquet('{filepath}')")


if __name__ == "__main__":
    main()
