from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_summary_tables(
    summaries: dict[str, pd.DataFrame],
    output_folder: Path,
) -> None:
    """Save the four required summary tables."""

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    summaries["grouped"].to_csv(
        output_folder / "grouped.csv",
        index=False
    )

    summaries["grouped_two"].to_csv(
        output_folder / "grouped_two.csv",
        index=False
    )

    summaries["pivot"].to_csv(
        output_folder / "pivot.csv",
        index=False
    )

    summaries["top10"].to_csv(
        output_folder / "top10.csv",
        index=False
    )


def create_bar_chart(
    top10: pd.DataFrame,
    output_folder: Path,
) -> None:
    """Create a bar chart of the ten largest country groups."""

    plot_data = top10.sort_values(
        "measure_sum",
        ascending=True
    )

    plt.figure(figsize=(12, 7))

    plt.barh(
        plot_data["countryorigin_iso3"],
        plot_data["measure_sum"]
    )

    plt.title(
        "Top 10 Countries by Dutiable Value - Customs 2015"
    )

    plt.xlabel(
        "Dutiable Value (PHP)"
    )

    plt.ylabel(
        "Country of Origin (ISO3)"
    )

    plt.tight_layout()

    plt.savefig(
        output_folder / "bar.png",
        dpi=150
    )

    plt.close()


def create_heatmap(
    pivot: pd.DataFrame,
    output_folder: Path,
) -> None:
    """Create a heatmap of country and transaction quarter."""

    heatmap_data = pivot[
        pivot["countryorigin_iso3"] != "Total"
    ].copy()

    heatmap_data = heatmap_data.set_index(
        "countryorigin_iso3"
    )

    if "Total" in heatmap_data.columns:
        heatmap_data = heatmap_data.drop(
            columns=["Total"]
        )

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap="Blues",
        fmt=".0f"
    )

    plt.title(
        "Dutiable Value by Country of Origin and Quarter"
    )

    plt.xlabel(
        "Transaction Quarter"
    )

    plt.ylabel(
        "Country of Origin (ISO3)"
    )

    plt.tight_layout()

    plt.savefig(
        output_folder / "heatmap.png",
        dpi=150
    )

    plt.close()