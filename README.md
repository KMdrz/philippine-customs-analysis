# Philippine Customs 2015 Data Summary Program

## 1. Project Description

This project analyzes the Philippine Customs 2015 dataset from BetterGov.PH.

The program reads the Customs dataset, filters records, creates derived variables, generates summary tables and visualizations, performs NumPy analysis, and validates the resulting totals.

---

## 2. Dataset

Dataset:
Philippine Customs 2015

Filename:
2015.csv

Source:
BetterGov.PH Open Customs Data

File size:
Approximately 493.5 MB

Place the downloaded file here:

data/2015.csv

The raw dataset is not included in this repository.

---

## 3. Selected Fields

| Field | Meaning | Unit |
|---|---|---|
| countryorigin_iso3 | Country of origin represented by ISO3 code | categorical |
| tq | Transaction quarter | categorical |
| dutiablevaluephp | Dutiable value | Philippine pesos |

---

## 4. Filtering Rule

Records are selected when:

tq is not missing

AND

dutiablevaluephp > PHP 1,000,000

Records with missing filter values are placed in the excluded group.

---

## 5. Derived Columns

### dutiablevalue_million_php

Converts dutiable value from Philippine pesos to millions of Philippine pesos.

### high_value_flag

Records with dutiable value greater than or equal to PHP 10,000,000 are classified as "High"; other selected records are classified as "Regular".

---

## 6. Outputs

### grouped.csv

Groups records by country of origin and reports row count, valid measure count, sum, and mean.

### grouped_two.csv

Groups records by country of origin and transaction quarter.

### pivot.csv

Shows dutiable value sums across country of origin and transaction quarter using a pivot table.

### top10.csv

Contains the ten country groups with the largest dutiable value sums.

### bar.png

The bar chart compares the ten country groups with the largest total dutiable value.

### heatmap.png

The heatmap shows dutiable value across country-of-origin groups and transaction quarters, excluding pivot margins.

### validation.csv

Contains validation checks comparing expected and actual values.

### audit_log.csv

Records important processing operations and row counts.

### numpy_timing.csv

Contains timing results for loop and vectorized NumPy calculations.

---

## 7. Installation

Install the required packages:

pip install -r requirements.txt

---

## 8. Running the Program

Place 2015.csv in:

data/2015.csv

Then run:

python main.py

---

## 9. Validation Reference

Customs 2015 reference:

Raw rows:
2,236,612

Raw dutiable value:
PHP 3,587,267,375,257

The validation uses an absolute tolerance of PHP 1.00 and zero relative tolerance.

---

## 10. Missing Values

Missing numerical values are reported separately and are not replaced with zero.

Missing categorical values used in the selected records are represented explicitly as [Missing].

---

## 11. Git Workflow

Each member developed their assigned work on a separate branch.

Each contribution was reviewed by another member through GitHub.

Reviewed branches were merged into main.

The final submitted version is tagged:

week6-v1.0