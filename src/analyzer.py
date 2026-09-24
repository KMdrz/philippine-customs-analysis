from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class CustomsAnalyzer:
    """Analyze the Philippine Customs 2015 CSV dataset."""

    def __init__(
        self,
        input_path: Path,
        output_folder: Path,
        group_columns: list[str],
        measure_column: str,
        chunk_size: int = 100_000,
    ) -> None:
        """Initialize the analyzer with project settings."""

        self.input_path = input_path
        self.output_folder = output_folder
        self.group_columns = group_columns
        self.measure_column = measure_column
        self.chunk_size = chunk_size

        self.raw_rows = 0
        self.raw_sum = 0.0
        self.selected_rows = 0
        self.excluded_rows = 0

        self.selected_data = pd.DataFrame()
        self.grouped = pd.DataFrame()
        self.grouped_two = pd.DataFrame()
        self.pivot = pd.DataFrame()
        self.top10 = pd.DataFrame()

        self.audit_log: list[dict[str, Any]] = []

    def inspect_columns(self) -> dict[str, Any]:
        """Inspect columns, data types, and missing values."""

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.input_path}"
            )

        header = pd.read_csv(
            self.input_path,
            nrows=0,
            encoding="latin1",
        )

        sample = pd.read_csv(
            self.input_path,
            nrows=5000,
            low_memory=False,
            encoding="latin1",
        )

        inspection = {
            "columns": list(header.columns),
            "column_count": len(header.columns),
            "dtypes": sample.dtypes.astype(str).to_dict(),
            "missing_values": sample.isna().sum().to_dict(),
        }

        print("\n--- DATASET INSPECTION ---")

        print(
            f"Columns: {inspection['column_count']}"
        )

        print("\nData types:")
        print(sample.dtypes)

        print("\nMissing values in sample:")
        print(sample.isna().sum())

        return inspection

    def filter_and_transform(
        self,
        filter_quarter: str | None = None,
        minimum_value_php: float = 1_000_000,
    ) -> pd.DataFrame:
        """Filter records and create two derived columns."""

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.input_path}"
            )

        selected_chunks = []

        for chunk in pd.read_csv(
            self.input_path,
            usecols=[
                "countryorigin_iso3",
                "tq",
                "dutiablevaluephp",
            ],
            chunksize=self.chunk_size,
            low_memory=False,
            encoding="latin1",
        ):
            chunk["dutiablevaluephp"] = pd.to_numeric(
                chunk["dutiablevaluephp"],
                errors="coerce",
            )

            self.raw_rows += len(chunk)

            self.raw_sum += chunk[
                "dutiablevaluephp"
            ].sum(
                skipna=True
            )

            if filter_quarter is None:
                mask = (
                    chunk["tq"].notna()
                    & (
                        chunk["dutiablevaluephp"]
                        > minimum_value_php
                    )
                )

            else:
                mask = (
                    (chunk["tq"] == filter_quarter)
                    & (
                        chunk["dutiablevaluephp"]
                        > minimum_value_php
                    )
                )

            filtered = chunk.loc[mask].copy()

            if not filtered.empty:
                selected_chunks.append(
                    filtered
                )

            self.selected_rows += len(filtered)

        self.excluded_rows = (
            self.raw_rows
            - self.selected_rows
        )

        if self.selected_rows == 0:
            raise ValueError(
                "The filter returned no rows. "
                "Check the filter settings."
            )

        selected = pd.concat(
            selected_chunks,
            ignore_index=True,
        )

        # Derived numerical column.
        selected[
            "dutiablevalue_million_php"
        ] = (
            selected["dutiablevaluephp"]
            / 1_000_000
        )

        # Derived categorical flag.
        selected["high_value_flag"] = np.where(
            selected["dutiablevaluephp"]
            >= 10_000_000,
            "High",
            "Regular",
        )

        # Missing category values are represented explicitly.
        selected[
            "countryorigin_iso3"
        ] = (
            selected["countryorigin_iso3"]
            .fillna("[Missing]")
            .astype(str)
        )

        selected["tq"] = (
            selected["tq"]
            .fillna("[Missing]")
            .astype(str)
        )

        self.selected_data = selected

        self.audit_log.append({
            "step": "filter",
            "operation": "Apply two-condition filter",
            "rule": (
                "tq is not missing AND "
                "dutiablevaluephp > "
                f"{minimum_value_php}"
            ),
            "rows_before": self.raw_rows,
            "rows_after": self.selected_rows,
        })

        return selected

    def create_summaries(
        self,
    ) -> dict[str, pd.DataFrame]:
        """Create the required summary tables."""

        if self.selected_data.empty:
            raise ValueError(
                "No selected data available."
            )

        data = self.selected_data

        self.grouped = (
            data.groupby(
                "countryorigin_iso3",
                dropna=False,
            )
            .agg(
                row_count=(
                    "dutiablevaluephp",
                    "size",
                ),
                valid_measure_count=(
                    "dutiablevaluephp",
                    "count",
                ),
                measure_sum=(
                    "dutiablevaluephp",
                    "sum",
                ),
                measure_mean=(
                    "dutiablevaluephp",
                    "mean",
                ),
            )
            .reset_index()
            .sort_values(
                "measure_sum",
                ascending=False,
            )
        )

        self.grouped_two = (
            data.groupby(
                [
                    "countryorigin_iso3",
                    "tq",
                ],
                dropna=False,
            )
            .agg(
                row_count=(
                    "dutiablevaluephp",
                    "size",
                ),
                measure_sum=(
                    "dutiablevaluephp",
                    "sum",
                ),
            )
            .reset_index()
            .sort_values(
                "measure_sum",
                ascending=False,
            )
        )

        self.pivot = pd.pivot_table(
            data,
            index="countryorigin_iso3",
            columns="tq",
            values="dutiablevaluephp",
            aggfunc="sum",
            fill_value=0,
            margins=True,
            margins_name="Total",
        ).reset_index()

        self.top10 = (
            self.grouped
            .sort_values(
                "measure_sum",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        return {
            "grouped": self.grouped,
            "grouped_two": self.grouped_two,
            "pivot": self.pivot,
            "top10": self.top10,
        }

    def validate_results(
        self,
        expected_raw_rows: int,
        expected_raw_sum: float,
        absolute_tolerance: float = 1.00,
    ) -> pd.DataFrame:
        """Check raw totals and internal summary consistency."""

        if self.selected_data.empty:
            raise ValueError(
                "No selected data available for validation."
            )

        checks = []

        selected_sum = (
            self.selected_data[
                self.measure_column
            ].sum()
        )

        # Check the external Customs 2015 row reference.
        checks.append({
            "check": "Raw row count",
            "expected": expected_raw_rows,
            "actual": self.raw_rows,
            "tolerance": 0,
            "pass": (
                self.raw_rows
                == expected_raw_rows
            ),
        })

        # Check the external Customs 2015 measure reference.
        checks.append({
            "check": "Raw dutiable value sum",
            "expected": expected_raw_sum,
            "actual": self.raw_sum,
            "tolerance": absolute_tolerance,
            "pass": np.isclose(
                self.raw_sum,
                expected_raw_sum,
                atol=absolute_tolerance,
                rtol=0,
            ),
        })

        # Check that filtering accounts for every raw record.
        checks.append({
            "check": "Raw rows = selected + excluded",
            "expected": self.raw_rows,
            "actual": (
                self.selected_rows
                + self.excluded_rows
            ),
            "tolerance": 0,
            "pass": (
                self.raw_rows
                == self.selected_rows
                + self.excluded_rows
            ),
        })

        # Check grouped row counts against selected records.
        grouped_row_total = int(
            self.grouped[
                "row_count"
            ].sum()
        )

        checks.append({
            "check": "Grouped rows = selected rows",
            "expected": self.selected_rows,
            "actual": grouped_row_total,
            "tolerance": 0,
            "pass": (
                grouped_row_total
                == self.selected_rows
            ),
        })

        # Check grouped sums against the independent selected sum.
        grouped_sum = self.grouped[
            "measure_sum"
        ].sum()

        checks.append({
            "check": "Grouped sum = selected sum",
            "expected": selected_sum,
            "actual": grouped_sum,
            "tolerance": absolute_tolerance,
            "pass": np.isclose(
                grouped_sum,
                selected_sum,
                atol=absolute_tolerance,
                rtol=0,
            ),
        })

        # Exclude pivot margins before checking the pivot sum.
        pivot_without_margin = self.pivot[
            self.pivot[
                "countryorigin_iso3"
            ] != "Total"
        ]

        pivot_columns = [
            column
            for column in pivot_without_margin.columns
            if (
                column != "countryorigin_iso3"
                and column != "Total"
            )
        ]

        pivot_sum = (
            pivot_without_margin[
                pivot_columns
            ]
            .sum()
            .sum()
        )

        checks.append({
            "check": "Pivot interior sum = selected sum",
            "expected": selected_sum,
            "actual": pivot_sum,
            "tolerance": absolute_tolerance,
            "pass": np.isclose(
                pivot_sum,
                selected_sum,
                atol=absolute_tolerance,
                rtol=0,
            ),
        })

        # Confirm the Top 10 table comes from the grouped table.
        top10_sum = self.top10[
            "measure_sum"
        ].sum()

        expected_top10_sum = (
            self.grouped
            .head(10)[
                "measure_sum"
            ]
            .sum()
        )

        checks.append({
            "check": "Top10 values match grouped table",
            "expected": expected_top10_sum,
            "actual": top10_sum,
            "tolerance": absolute_tolerance,
            "pass": np.isclose(
                top10_sum,
                expected_top10_sum,
                atol=absolute_tolerance,
                rtol=0,
            ),
        })

        return pd.DataFrame(checks)