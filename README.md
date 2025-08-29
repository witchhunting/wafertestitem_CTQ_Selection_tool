# wafertestitem_CTQ_Selection_tool

CTQ Index Selection from wafer test item by fail die.

All processing lives in `ctq.py`. The script defines helper functions
for data generation, preprocessing and a vectorized point-biserial
correlation with bootstrap confidence intervals. ``process_map`` from
`tqdm` accelerates running the bootstrap across many parameters.

## Usage

1. Open `ctq.py` and set the options near the bottom of the file
   (e.g. `pos_label`, `bootstrap_iterations`, `corr_threshold`).
2. Run the script:

   ```bash
   python ctq.py
   ```
3. The script executes the steps sequentially:
   data generation → preprocessing → parallel correlation.
4. The output lists parameters whose correlation exceeds the chosen
   threshold. ``ci_low`` and ``ci_high`` show the lower and upper
   bounds of the 95% bootstrap confidence interval.

Replace the toy data generation and preprocessing functions with logic
for your dataset as needed.
