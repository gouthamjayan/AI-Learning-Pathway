# Linear Regression

Linear Regression predicts a continuous number by fitting a straight line (or flat plane in higher dimensions) through your data.

## How It Works
- Start with a guess for the weights and bias to make predictions.
- Measure how far off those predictions are from the actual values using error metrics like MSE, MAE, RMSE, and $R^2$.
- Adjust the weights and bias step by step so the error gets smaller.

## What's in This Folder
- **NumPy implementation** — Built from scratch so you can see how predictions, errors, and gradient updates actually work under the hood (including keeping shapes aligned like `(N, 1)` so broadcasting doesn't trip you up).
- **Standard implementation** — Uses scikit-learn for a clean, ready-to-use workflow.

## Good Things to Remember
- Check feature and target shapes early so shape mismatches don't cause silent bugs.
- Compare `R²` alongside `RMSE` or `MAE` to see both how well the line fits and how big typical mistakes are.