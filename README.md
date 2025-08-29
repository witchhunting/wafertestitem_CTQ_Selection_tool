# wafertestitem_CTQ_Selection_tool

CTQ Index Selection from wafer test item by fail die.

## Overview

This project provides a bootstrap point-biserial correlation analysis to identify critical-to-quality (CTQ) test items that relate to a specific failure type. The workflow is inspired by the high-performance data-processing examples from [QuantProject](https://github.com/xiangkunqin/QuantProject).

## Installation

```bash
pip install numpy pandas tqdm
```

## Usage

```python
import pandas as pd
from ctq_selector import pointbiserial_bootstrap

# Load your wafer test data
# df = pd.read_csv("wafer_test.csv")
# test_items = ["item1", "item2", ...]

results = pointbiserial_bootstrap(
    df,
    pos_label="M0C-GT Short",
    y_col="y_pFA",
    test_item_columns=test_items,
    n_bootstrap=1000,
)
print(results.head())
```

Set `save_prefix` to export full, top-50, and significant feature lists as CSV files.

## License

This project is distributed without a specific license.
