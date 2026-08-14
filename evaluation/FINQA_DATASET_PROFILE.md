# FinQA Dataset Profiling Report

This report is auto-generated from raw FinQA JSON files in `data/raw/FinQA-main/dataset`.

## Checklist Coverage

- [x] Explore FinQA dataset structure
- [x] Understand JSON schema (tables, text, questions, programs, answers)
- [x] Identify question types (percentage, difference, ratio, multi-hop)
- [x] Identify edge cases (missing values, merged cells, multi-year tables)
- [x] Document financial metric categories found in dataset

## JSON Schema Summary

### Top-level keys (intersection across splits)

- filename
- id
- post_text
- pre_text
- qa
- table
- table_ori

### QA keys (intersection across splits)

- question

## Split Statistics

### train

- Records: 6251
- Records with program: 6251
- Avg question token count: 16.39
- Avg table rows: 6.34
- Avg pre-text sentences: 11.59
- Avg post-text sentences: 12.68
- Top program ops:
  - divide: 4445
  - subtract: 2739
  - add: 1512
  - multiply: 567
  - greater: 124
  - table_average: 95
  - table_max: 48
  - table_sum: 36
- Question types:
  - ratio: 4422
  - difference: 2856
  - percentage: 2725
  - multi_hop: 2534
  - other: 561
- Financial metric categories:
  - ratio_margin: 3189
  - balance_sheet: 1415
  - cost_expense: 1381
  - shares_capital: 1236
  - revenue_sales: 1038
  - tax: 831
  - cash_flow_liquidity: 783
  - profitability_income: 766

### dev

- Records: 883
- Records with program: 883
- Avg question token count: 16.12
- Avg table rows: 6.27
- Avg pre-text sentences: 11.02
- Avg post-text sentences: 13.11
- Top program ops:
  - divide: 636
  - subtract: 416
  - add: 180
  - multiply: 83
  - table_average: 19
  - greater: 10
  - table_max: 8
  - table_min: 5
- Question types:
  - ratio: 621
  - difference: 428
  - percentage: 380
  - multi_hop: 360
  - other: 78
- Financial metric categories:
  - ratio_margin: 462
  - balance_sheet: 219
  - cost_expense: 192
  - shares_capital: 181
  - revenue_sales: 164
  - profitability_income: 115
  - tax: 109
  - cash_flow_liquidity: 84

### test

- Records: 1147
- Records with program: 1147
- Avg question token count: 16.52
- Avg table rows: 6.55
- Avg pre-text sentences: 12.22
- Avg post-text sentences: 12.49
- Top program ops:
  - divide: 820
  - subtract: 521
  - add: 260
  - multiply: 109
  - greater: 20
  - table_average: 15
  - table_max: 10
  - table_sum: 10
- Question types:
  - ratio: 814
  - difference: 530
  - percentage: 503
  - multi_hop: 493
  - other: 125
- Financial metric categories:
  - ratio_margin: 591
  - balance_sheet: 294
  - shares_capital: 249
  - cost_expense: 211
  - tax: 177
  - revenue_sales: 152
  - cash_flow_liquidity: 147
  - profitability_income: 106

### private_test

- Records: 919
- Records with program: 0
- Avg question token count: 15.56
- Avg table rows: 6.14
- Avg pre-text sentences: 12.03
- Avg post-text sentences: 11.89
- Top program ops:
- Question types:
  - other: 413
  - difference: 373
  - percentage: 238
  - ratio: 79
- Financial metric categories:
  - ratio_margin: 400
  - balance_sheet: 226
  - shares_capital: 214
  - cost_expense: 153
  - tax: 136
  - revenue_sales: 119
  - cash_flow_liquidity: 96
  - profitability_income: 76

## Edge Cases

- dev:boolean_answer: 10
- dev:multi_year_tables: 677
- dev:possible_merged_cell_rows: 304
- dev:rows_with_missing_like_cells: 341
- private_test:missing_program: 919
- private_test:multi_year_tables: 721
- private_test:possible_merged_cell_rows: 374
- private_test:rows_with_missing_like_cells: 413
- test:boolean_answer: 22
- test:multi_year_tables: 933
- test:possible_merged_cell_rows: 423
- test:rows_with_missing_like_cells: 491
- train:boolean_answer: 120
- train:multi_year_tables: 4828
- train:possible_merged_cell_rows: 2236
- train:rows_with_missing_like_cells: 2741

## Notes

- `private_test` has no gold programs by design (blind evaluation split).
- `possible_merged_cell_rows` is a heuristic: rows with blank first cell and populated trailing cells.
- Question type detection is heuristic and should be refined in the parser for training/eval tasks.
