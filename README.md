# CardBalance-AI-Python

This code provides a complete pipeline for building and deploying a balance forecasting model. It includes robust steps for creating a reusable workflow, ensuring stakeholders can evaluate and utilize results effectively.

![alt text](https://github.com/gaptab/CardBalance-AI-Python/blob/main/predictions_vs_actuals.png)


This project involves building a machine learning pipeline to forecast credit card balances and assess interest rate risks for a banking book. Here's a high-level overview of the process:

**Data Simulation**: A synthetic dataset is generated, mimicking real-world features such as balance, credit limit, transaction volume, and income.

**Data Preparation**: The dataset is split into training and testing sets. Features (inputs) are separated from the target variable (balance for the next month).

**Model Training**: A Random Forest Regressor is trained on the prepared data to learn patterns and make predictions about future balances.

**Evaluation**: The trained model is evaluated using metrics like Mean Squared Error (MSE) and R² score to measure accuracy and explainability.

**Visualization**: A scatter plot is created to compare predicted balances against actual values, helping stakeholders visually assess the model's performance.

**Saving Outputs**: Key components like the dataset, model, predictions, and evaluation metrics are saved to files for reproducibility and future use.

**Automation and Deployment**: The pipeline is automated, with potential integration for monitoring and validating the model over time.


This streamlined workflow enables stakeholders to make data-driven decisions, visualize model performance, and maintain comprehensive documentation for lifecycle management.
