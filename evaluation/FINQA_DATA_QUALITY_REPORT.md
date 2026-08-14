# FinQA Data Quality Report

Generated at: 2026-08-05T12:48:10.108890+00:00

## Validation Scope

- Parsed records validated against original raw JSON by record id
- Field parity checks for filename/question/answer/program
- Missing/null analysis and anomaly counts

## Split Results

### train

- Raw records: 6251
- Parsed records: 6251
- Coverage %: 100.0000
- Missing counts:
  - missing_answer: 48
- Missing percentages:
  - missing_answer_pct: 0.7679
- Quality percentages:
  - has_multi_year_table: 0.6978
  - table_missing_cell_pct: 2.7750
- Anomalies:
  - none

### dev

- Raw records: 883
- Parsed records: 883
- Coverage %: 100.0000
- Missing counts:
  - missing_answer: 12
- Missing percentages:
  - missing_answer_pct: 1.3590
- Quality percentages:
  - has_multi_year_table: 0.7010
  - table_missing_cell_pct: 2.2389
- Anomalies:
  - none

### test

- Raw records: 1147
- Parsed records: 1147
- Coverage %: 100.0000
- Missing counts:
  - missing_answer: 14
- Missing percentages:
  - missing_answer_pct: 1.2206
- Quality percentages:
  - has_multi_year_table: 0.6888
  - table_missing_cell_pct: 2.6802
- Anomalies:
  - none

### private_test

- Raw records: 919
- Parsed records: 919
- Coverage %: 100.0000
- Missing counts:
  - missing_answer: 919
  - missing_program: 919
- Missing percentages:
  - missing_answer_pct: 100.0000
  - missing_program_pct: 100.0000
- Quality percentages:
  - has_multi_year_table: 0.6551
  - table_missing_cell_pct: 2.8187
- Anomalies:
  - none
