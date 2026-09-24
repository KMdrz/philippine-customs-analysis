import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG, REQUIRED_COLUMNS
from src.analyzer import CustomsAnalyzer
from src.outputs import (
    create_bar_chart,
    create_heatmap,
    save_summary_tables,
)


def check_required_columns(
    input_path: Path,
    required_columns: set[str],
) -> None:
    """Verify that all required columns exist."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    header = pd.read_csv(
        input_path,
        nrows=0
    )

    actual_columns = set(header.columns)

    missing_columns = (
        required_columns
        - actual_columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )


def compare_numpy_methods(
    values: np.ndarray,
    repetitions: int = 5,
) -> pd.DataFrame:
    """Compare loop and vectorized NumPy calculations."""

    loop_times = []
    vectorized_times = []

    loop_result = None
    vectorized_result = None

    for _ in range(repetitions):

        start = time.perf_counter()

        loop_result = np.array([
            value / 1_000_000
            for value in values
        ])

        loop_times.append(
            time.perf_counter() - start
        )

        start = time.perf_counter()

        vectorized_result = (
            values / 1_000_000
        )

        vectorized_times.append(
            time.perf_counter() - start
        )

    equal_results = np.allclose(
        loop_result,
        vectorized_result
    )

    return pd.DataFrame({
        "method": [
            "Loop",
            "Vectorized"
        ],
        "median_time_seconds": [
            np.median(loop_times),
            np.median(vectorized_times),
        ],
        "repetitions": [
            repetitions,
            repetitions,
        ],
        "results_equal": [
            equal_results,
            equal_results,
        ],
    })


def create_numpy_demo(
    selected_data: pd.DataFrame,
    output_folder: Path,
    seed: int = 42,
    sample_size: int = 1000,
) -> pd.DataFrame:
    """Demonstrate NumPy arrays, masks, vectorization, and aggregation."""

    values = (
        selected_data["dutiablevaluephp"]
        .dropna()
        .to_numpy(dtype=float)
    )

    rng = np.random.default_rng(seed)

    actual_sample_size = min(
        sample_size,
        len(values)
    )

    sample_indices = rng.choice(
        len(values),
        size=actual_sample_size,
        replace=False
    )

    sample = values[sample_indices]

    # Convert the selected values into a NumPy array.
    values_array = np.asarray(sample)

    # Create a Boolean mask for values above PHP 1,000,000.
    mask = values_array > 1_000_000

    # Apply the Boolean mask.
    masked_values = values_array[mask]

    # Perform a vectorized unit conversion.
    values_in_millions = (
        masked_values / 1_000_000
    )

    # Aggregate the vectorized results.
    total_millions = np.sum(
        values_in_millions
    )

    print("\n--- NUMPY DEMONSTRATION ---")

    print(
        f"Sample size: {len(values_array)}"
    )

    print(
        f"Values above PHP 1,000,000: "
        f"{len(masked_values)}"
    )

    print(
        f"Total in million PHP: "
        f"{total_millions:,.2f}"
    )

    timing = compare_numpy_methods(
        values_array
    )

    timing.to_csv(
        output_folder / "numpy_timing.csv",
        index=False
    )

    print("\nLoop vs vectorized:")
    print(timing)

    if not timing["results_equal"].all():
        raise ValueError(
            "Loop and vectorized results do not agree."
        )

    return timing


def main() -> int:
    """Run the complete Customs analysis."""

    print("=" * 60)
    print(
        "PHILIPPINE CUSTOMS 2015 "
        "DATA SUMMARY PROGRAM"
    )
    print("=" * 60)

    input_path = CONFIG["input_path"]
    output_folder = CONFIG["output_folder"]

    try:

        # 1. Verify the input structure.
        print(
            "\n[1/7] Checking required columns..."
        )

        check_required_columns(
            input_path,
            REQUIRED_COLUMNS
        )

        print(
            "Required columns found."
        )

        # 2. Create the analysis object.
        analyzer = CustomsAnalyzer(
            input_path=input_path,
            output_folder=output_folder,
            group_columns=CONFIG[
                "group_columns"
            ],
            measure_column=CONFIG[
                "measure_column"
            ],
            chunk_size=CONFIG[
                "chunk_size"
            ],
        )

        # 3. Inspect the dataset.
        print(
            "\n[2/7] Inspecting dataset..."
        )

        analyzer.inspect_columns()

        # 4. Filter and transform the records.
        print(
            "\n[3/7] Filtering and transforming..."
        )

        analyzer.filter_and_transform(
            filter_quarter=CONFIG[
                "filter_quarter"
            ],
            minimum_value_php=CONFIG[
                "minimum_value_php"
            ],
        )

        print(
            f"Raw rows: {analyzer.raw_rows:,}"
        )

        print(
            f"Selected rows: "
            f"{analyzer.selected_rows:,}"
        )

        print(
            f"Excluded rows: "
            f"{analyzer.excluded_rows:,}"
        )

        print(
            f"Raw dutiable value: "
            f"PHP {analyzer.raw_sum:,.2f}"
        )

        # 5. Create summary tables.
        print(
            "\n[4/7] Creating summaries..."
        )

        summaries = (
            analyzer.create_summaries()
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        save_summary_tables(
            summaries,
            output_folder
        )

        # 6. Create required plots.
        print(
            "\n[5/7] Creating plots..."
        )

        create_bar_chart(
            analyzer.top10,
            output_folder
        )

        create_heatmap(
            analyzer.pivot,
            output_folder
        )

        # 7. Run NumPy demonstration.
        print(
            "\n[6/7] Running NumPy..."
        )

        create_numpy_demo(
            analyzer.selected_data,
            output_folder,
            seed=CONFIG["random_seed"],
            sample_size=CONFIG[
                "numpy_sample_size"
            ],
        )

        print(
            "\nMain analysis completed."
        )

        return 0

    except FileNotFoundError as error:

        print(
            f"\nERROR: {error}"
        )

        return 1

    except ValueError as error:

        print(
            f"\nERROR: {error}"
        )

        return 1

    except Exception as error:

        print(
            "\nUNEXPECTED ERROR: "
            f"{type(error).__name__}: {error}"
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())