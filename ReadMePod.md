# Kubernetes Pod Failure Prediction Model

## Overview
This project implements a machine learning pipeline to predict Kubernetes pod failures before they occur. By analyzing pod status, metrics, and log data, we aim to identify potential issues proactively, allowing for timely intervention and improved cluster stability.

## Key Features
- Data preprocessing and feature engineering from Kubernetes pod metrics and logs
- Text feature extraction from pod logs using TF-IDF
- Model training using Decision Tree and Random Forest classifiers
- Comprehensive model evaluation and visualization
- Feature importance analysis for interpretability
- Docker integration for deployment and testing

## Approach and Rationale

### Data Collection and Preprocessing
We used a synthetic dataset (`synthetic_pod_status.csv`) to simulate real-world Kubernetes pod data. Our preprocessing steps include:
1. Handling missing values in log data
2. Removing duplicates to prevent data leakage
3. Encoding categorical variables (pod status and failure reasons)
4. Applying TF-IDF to extract features from log text

This approach allows us to capture both structured metrics and unstructured log data, providing a comprehensive view of pod health.

### Feature Engineering
We focused on creating a rich feature set that combines:
- Numeric pod metrics (e.g., restart count)
- Encoded categorical data (status and failure reasons)
- Text-based features from logs (using TF-IDF)

This multi-modal approach enables the model to learn from various aspects of pod behavior and performance.

### Model Selection
We chose Decision Trees and Random Forests for their:
1. Interpretability: Easy to understand which features contribute most to predictions
2. Handling of mixed data types: Can work well with both categorical and numerical features
3. Robustness to outliers and non-linear relationships

Random Forests, in particular, help reduce overfitting through ensemble learning.

### Evaluation Metrics
We use a comprehensive set of evaluation metrics:
- Accuracy and classification report for overall performance
- ROC curves and AUC scores for binary classification performance
- Precision-Recall curves to handle potential class imbalance
- Cross-validation to assess model generalization

### Feature Importance Analysis
We leverage Random Forest's feature importance to identify key predictors of pod failures. This provides valuable insights for cluster administrators and developers.

### Docker Integration
Docker integration allows for:
1. Consistent environment for model deployment
2. Easy testing with simulated log data
3. Potential for scaling and integration with Kubernetes clusters

## Usage
1. Ensure Docker is installed and running on your system.
2. Load the Docker image:
   ```
   docker load < fake-docker.tar
   ```
3. Run the Docker container:
   ```
   docker run -d --name fake-docker-container k8s-failure-prediction
   ```
4. Access the prediction interface:
   ```
   docker exec -it fake-docker-container python predict.py
   ```
5. View logs:
   ```
   docker logs fake-docker-container
   ```
6. Retrieve results:
   ```
   docker cp fake-docker-container:/app/logs/results.txt ./
   ```

## Future Improvements
1. Real-time prediction integration with Kubernetes clusters
2. Incorporation of time-series analysis for temporal patterns
3. Exploration of deep learning models for more complex pattern recognition
4. Automated model retraining and updating based on new data

## Conclusion
This project demonstrates a practical approach to predicting Kubernetes pod failures using machine learning. By combining structured metrics with unstructured log data, we create a robust prediction model that can potentially improve cluster reliability and reduce downtime.
