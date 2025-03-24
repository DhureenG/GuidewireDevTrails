import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("/Users/dhureengulati/Downloads/synthetic_pod_status.csv")
df["Logs"] = df["Logs"].fillna("No logs available")  # Fill NaN values
# Telling pandas to display all columns and not just skip some with "..."
pd.set_option('display.max_columns', None)
# Telling pandas to display all columns to print in one row and not break it apart
pd.set_option('display.expand_frame_repr', False)
print(df.head)
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["Failure_Label"].value_counts(normalize=True) * 100)

# Verify data for target leakage
print("\nChecking for potential data leakage...")
correlation = df.select_dtypes(include=[np.number]).corr()
if 'Failure_Label' in correlation.columns:
    high_corr_features = correlation['Failure_Label'][
        (correlation['Failure_Label'] > 0.8) | (correlation['Failure_Label'] < -0.8)
    ].index.tolist()
    high_corr_features.remove('Failure_Label') if 'Failure_Label' in high_corr_features else None
    
    if high_corr_features:
        print(f"⚠️ Warning: Features with high correlation to target: {high_corr_features}")
        print("Consider removing these to prevent data leakage")
    else:
        print("✅ No numerical features with suspiciously high correlation to target")

# Add noise to make the problem more realistic (if needed)
def add_noise_to_dataset(df, noise_level=0.05):
    """Add noise to make the classification task more challenging"""
    # Clone the dataframe
    df_noisy = df.copy()
    
    # Randomly flip some labels
    mask = np.random.random(len(df)) < noise_level
    df_noisy.loc[mask, 'Failure_Label'] = 1 - df_noisy.loc[mask, 'Failure_Label']
    
    print(f"✅ Added noise by flipping {mask.sum()} labels ({noise_level*100:.1f}% of data)")
    return df_noisy

# Uncomment to add noise
df = add_noise_to_dataset(df, noise_level=0.05)

# Feature Engineering
print("\nPerforming feature engineering...")
tfidf = TfidfVectorizer(max_features=100, min_df=5, max_df=0.7, ngram_range=(1, 2))
log_features = tfidf.fit_transform(df["Logs"])
log_features_df = pd.DataFrame(log_features.toarray(), columns=[f"log_tfidf_{i}" for i in range(log_features.shape[1])])

# Create feature set excluding any direct indicators of failure
X = pd.concat([
    pd.get_dummies(df["Status"], drop_first=True, prefix='status'),
    df[["Restart Count"]],
    pd.get_dummies(df["Failure Reasons"], drop_first=True, dummy_na=True, prefix='reason'),
    log_features_df
], axis=1)

y = df["Failure_Label"]
X.columns = X.columns.astype(str)  # Ensure all feature names are strings

# Normalize numerical features
scaler = StandardScaler()
X['Restart Count'] = scaler.fit_transform(X[['Restart Count']])

print(f"Final feature set: {X.shape[1]} features")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Function to plot ROC curve
def plot_roc_curve(model, X_test, y_test, name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{name}_roc_curve.png")
    return roc_auc

# Function to plot PR curve for class imbalance
def plot_pr_curve(model, X_test, y_test, name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{name}_pr_curve.png")
    return avg_precision

# Function to plot learning curve
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=StratifiedKFold(n_splits=5), 
        scoring='f1', train_sizes=np.linspace(0.1, 1.0, 10)
    )
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.grid(True, alpha=0.3)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    
    plt.legend(loc="best")
    plt.savefig(f"{title.replace(' ', '_')}.png")

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{name}_confusion_matrix.png")

# Function to evaluate models with cross-validation
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='f1')
    print(f"\n📌 {name} Model Evaluation:")
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train on full training set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Basic metrics
    print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
    print(classification_report(y_test, y_pred))
    
    # ROC curve
    roc_auc = plot_roc_curve(model, X_test, y_test, name)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # PR curve
    avg_precision = plot_pr_curve(model, X_test, y_test, name)
    print(f"Average Precision Score: {avg_precision:.4f}")
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, name)
    
    # Permutation feature importance (more reliable than model's built-in)
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': perm_importance.importances_mean
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20), hue='Feature', legend=False)
    plt.title(f'{name} - Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(f"{name}_feature_importance.png")
    
    return feature_importance['Feature'].tolist()

# Train and evaluate models
print("\n🔄 Training and evaluating models...")
dt_model = DecisionTreeClassifier(
    max_depth=8, 
    min_samples_split=10, 
    min_samples_leaf=5, 
    class_weight='balanced', 
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=12, 
    min_samples_split=10, 
    min_samples_leaf=5, 
    max_features='sqrt', 
    class_weight='balanced', 
    random_state=42
)

# Plot learning curves
plot_learning_curve(dt_model, X_train, y_train, "Decision Tree Learning Curve")
plot_learning_curve(rf_model, X_train, y_train, "Random Forest Learning Curve")

# Evaluate models
top_dt_features = evaluate_model(dt_model, X_train, X_test, y_train, y_test, "Decision Tree")
top_rf_features = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")

# Feature selection and retraining
if top_rf_features and top_dt_features:
    common_features = list(set(top_rf_features[:20]).intersection(set(top_dt_features[:20])))
    
    if common_features:
        print(f"\n✅ Using {len(common_features)} common important features for retraining")
        X_selected = X[common_features]
        X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
            X_selected, y, test_size=0.2, stratify=y, random_state=42
        )

        rf_model_reduced = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=10, 
            min_samples_leaf=5, 
            max_features='sqrt', 
            class_weight='balanced', 
            random_state=42
        )
        
        plot_learning_curve(rf_model_reduced, X_train_sel, y_train_sel, "Random Forest (Reduced Features) Learning Curve")
        evaluate_model(rf_model_reduced, X_train_sel, X_test_sel, y_train_sel, y_test_sel, "Random Forest (Reduced Features)")

print("\n✅ Model evaluation completed. Check generated plots for insights.")