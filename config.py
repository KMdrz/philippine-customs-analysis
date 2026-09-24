from pathlib import Path


CONFIG = {
    "input_path": Path("data/2015.csv"),
    "output_folder": Path("outputs"),

    "group_columns": [
        "countryorigin_iso3",
        "tq"
    ],

    "measure_column": "dutiablevaluephp",

    # Filter settings
    "filter_quarter": None,
    "minimum_value_php": 1_000_000,

    # Processing
    "chunk_size": 100_000,

    # NumPy
    "random_seed": 42,
    "numpy_sample_size": 1000,

    # Customs 2015 reference values
    "expected_raw_rows": 2_236_612,
    "expected_raw_sum": 3_587_267_375_257.0,

    "absolute_tolerance": 1.00,
    "relative_tolerance": 0.0,
}


REQUIRED_COLUMNS = {
    "countryorigin_iso3",
    "tq",
    "dutiablevaluephp",
}