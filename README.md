# wafertestitem_CTQ_Selection_tool

CTQ Index Selection from wafer test item by fail die.

All processing lives in a single script `ctq.py`. The script defines
helper functions for data generation, preprocessing and a vectorized
point-biserial correlation with bootstrap confidence intervals.

## Usage

1. Open `ctq.py` and set the options near the bottom of the file
   (e.g. `pos_label`, number of bootstrap iterations, etc.).
2. Run the script:

   ```bash
   python ctq.py
   ```
3. The script executes the steps sequentially:
   data generation → preprocessing → correlation.

Replace the toy data generation and preprocessing functions with logic
for your dataset as needed.
