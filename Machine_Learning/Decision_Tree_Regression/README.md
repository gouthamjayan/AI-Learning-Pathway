# Decision Tree

A Decision Tree makes predictions by asking a series of simple if/else questions about your features until it reaches an answer[cite: 3].

## How It Works
- Look through your features and find a split rule that groups similar data points together.
- Branch left or right based on whether each sample passes the rule.
- Repeat the process down the tree until reaching a leaf node that makes the final prediction[cite: 3].

## What's in This Folder
- **`decision_tree_reg.py`** — Loads the California Housing dataset, explores and cleans the data, splits into train and test sets, trains a `DecisionTreeRegressor`, and evaluates predictions using `R²` and `RMSE`[cite: 3].

## Good Things to Remember
- Watch out for overfitting when a tree grows too deep (like `max_depth=30`)—it can memorize training details instead of generalizing[cite: 3].
- Always check data shapes, duplicates, and summary stats before training so pipeline steps stay clean[cite: 3].