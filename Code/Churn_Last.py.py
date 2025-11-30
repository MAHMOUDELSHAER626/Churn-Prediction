import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                             recall_score, classification_report, roc_curve, 
                             roc_auc_score, precision_recall_curve)
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv(r"WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# =============================================================================
# 2. TARGET VARIABLE ANALYSIS
# =============================================================================
churn_numeric = {'Yes': 1, 'No': 0}
df.Churn.replace(churn_numeric, inplace=True)

# =============================================================================
# 3. ENHANCED EXPLORATORY DATA ANALYSIS
# =============================================================================

# 3.1 Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df.Churn.value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Churn Distribution (Count)', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Churn (0=No, 1=Yes)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['No Churn', 'Churn'], rotation=0)

df.Churn.value_counts(normalize=True).plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                             colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Churn Distribution (Percentage)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig('01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.2 Binary Features Analysis
binary_cols = []
for col in df.columns:
    if df[col].nunique() == 2:
        binary_cols.append(col)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for idx, col in enumerate(['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']):
    sns.countplot(x=col, hue='Churn', data=df, ax=axes[idx], palette=['#2ecc71', '#e74c3c'])
    axes[idx].set_title(f'{col} vs Churn', fontweight='bold')
    axes[idx].legend(title='Churn', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('02_binary_features_vs_churn.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.3 Churn Rate by Binary Features
binary_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
churn_rates = []
for col in binary_features:
    rate = df.groupby(col)['Churn'].mean()
    churn_rates.append(rate)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for idx, col in enumerate(binary_features):
    rate = df.groupby(col)['Churn'].mean()
    rate.plot(kind='bar', ax=axes[idx], color='#3498db')
    axes[idx].set_title(f'Churn Rate by {col}', fontweight='bold')
    axes[idx].set_ylabel('Churn Rate')
    axes[idx].set_ylim(0, 1)
    for container in axes[idx].containers:
        axes[idx].bar_label(container, fmt='%.2f')
plt.tight_layout()
plt.savefig('03_churn_rates_binary.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.4 Internet Service Analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(x="InternetService", data=df, ax=axes[0], palette='Set2')
axes[0].set_title('Internet Service Distribution', fontweight='bold')

sns.countplot(x="InternetService", hue='Churn', data=df, ax=axes[1], palette=['#2ecc71', '#e74c3c'])
axes[1].set_title('Internet Service vs Churn', fontweight='bold')
axes[1].legend(title='Churn', labels=['No', 'Yes'])

churn_by_internet = df.groupby('InternetService')['Churn'].mean()
churn_by_internet.plot(kind='bar', ax=axes[2], color='#e74c3c')
axes[2].set_title('Churn Rate by Internet Service', fontweight='bold')
axes[2].set_ylabel('Churn Rate')
for container in axes[2].containers:
    axes[2].bar_label(container, fmt='%.2f')
plt.tight_layout()
plt.savefig('04_internet_service_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.5 Additional Services Analysis
service_cols = ['StreamingTV', 'StreamingMovies', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for idx, col in enumerate(service_cols):
    churn_rate = df.groupby(col)['Churn'].mean()
    churn_rate.plot(kind='bar', ax=axes[idx], color='#9b59b6')
    axes[idx].set_title(f'Churn Rate by {col}', fontweight='bold')
    axes[idx].set_ylabel('Churn Rate')
    axes[idx].tick_params(axis='x', rotation=45)
    for container in axes[idx].containers:
        axes[idx].bar_label(container, fmt='%.2f', fontsize=8)
plt.tight_layout()
plt.savefig('05_services_churn_rates.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.6 Contract and Payment Method
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.countplot(x="Contract", data=df, ax=axes[0, 0], palette='Set3')
axes[0, 0].set_title('Contract Type Distribution', fontweight='bold')

sns.countplot(x="Contract", hue='Churn', data=df, ax=axes[0, 1], palette=['#2ecc71', '#e74c3c'])
axes[0, 1].set_title('Contract Type vs Churn', fontweight='bold')
axes[0, 1].legend(title='Churn', labels=['No', 'Yes'])

sns.countplot(x="PaymentMethod", data=df, ax=axes[1, 0], palette='Set3')
axes[1, 0].set_title('Payment Method Distribution', fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)

sns.countplot(x="PaymentMethod", hue='Churn', data=df, ax=axes[1, 1], palette=['#2ecc71', '#e74c3c'])
axes[1, 1].set_title('Payment Method vs Churn', fontweight='bold')
axes[1, 1].legend(title='Churn', labels=['No', 'Yes'])
axes[1, 1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('06_contract_payment_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.7 Numerical Features Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
sns.histplot(df["tenure"], kde=True, ax=axes[0, 0], color='#3498db')
axes[0, 0].set_title('Tenure Distribution', fontweight='bold')

sns.histplot(df["MonthlyCharges"], kde=True, ax=axes[0, 1], color='#e67e22')
axes[0, 1].set_title('Monthly Charges Distribution', fontweight='bold')

sns.histplot(data=df, x="tenure", hue="Churn", kde=True, ax=axes[0, 2], palette=['#2ecc71', '#e74c3c'])
axes[0, 2].set_title('Tenure by Churn', fontweight='bold')

sns.boxplot(x='Churn', y='tenure', data=df, ax=axes[1, 0], palette=['#2ecc71', '#e74c3c'])
axes[1, 0].set_title('Tenure Distribution by Churn', fontweight='bold')
axes[1, 0].set_xticklabels(['No Churn', 'Churn'])

sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[1, 1], palette=['#2ecc71', '#e74c3c'])
axes[1, 1].set_title('Monthly Charges by Churn', fontweight='bold')
axes[1, 1].set_xticklabels(['No Churn', 'Churn'])

sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df, ax=axes[1, 2], 
                palette=['#2ecc71', '#e74c3c'], alpha=0.5)
axes[1, 2].set_title('Tenure vs Monthly Charges', fontweight='bold')
plt.tight_layout()
plt.savefig('07_numerical_features_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.8 Correlation Heatmap (for numerical features)
numerical_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', linewidths=1, square=True)
plt.title('Correlation Heatmap - Numerical Features', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('08_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Drop uninformative columns
df.drop(['customerID', 'gender', 'PhoneService', 'Contract', 'TotalCharges'], axis=1, inplace=True)

# =============================================================================
# 4. DATA PREPROCESSING
# =============================================================================
cat_features = ['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']

X = pd.get_dummies(df, columns=cat_features, drop_first=True)
sc = MinMaxScaler()
X['tenure'] = sc.fit_transform(df[['tenure']])
X['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']])

# =============================================================================
# 5. RESAMPLING (UPSAMPLING MINORITY CLASS)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
X['Churn'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Class Distribution Before Resampling', fontweight='bold')
axes[0].set_xticklabels(['No Churn', 'Churn'], rotation=0)

X_no = X[X.Churn == 0]
X_yes = X[X.Churn == 1]
X_yes_upsampled = X_yes.sample(n=len(X_no), replace=True, random_state=42)
X_upsampled = pd.concat([X_no, X_yes_upsampled], axis=0).reset_index(drop=True)

X_upsampled['Churn'].value_counts().plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
axes[1].set_title('Class Distribution After Resampling', fontweight='bold')
axes[1].set_xticklabels(['No Churn', 'Churn'], rotation=0)
plt.tight_layout()
plt.savefig('09_resampling_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. TRAIN-TEST SPLIT
# =============================================================================
X = X_upsampled.drop(['Churn'], axis=1)
y = X_upsampled['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# =============================================================================
# 7. MODEL TRAINING WITH GRID SEARCH
# =============================================================================
results = {}
trained_models = {}

print("\n" + "="*80)
print("TRAINING MODELS WITH GRID SEARCH CV")
print("="*80)

# Logistic Regression
print("\n[1/5] Training Logistic Regression...")
log_parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000]
}
log_model = LogisticRegression(random_state=42, class_weight='balanced')
log_clf = GridSearchCV(estimator=log_model, param_grid=log_parameters, n_jobs=-1, cv=5, scoring='f1')
log_clf.fit(X, y)
y_pred_log = log_clf.predict(X_test)
y_pred_proba_log = log_clf.predict_proba(X_test)[:, 1]
cm_log = confusion_matrix(y_test, y_pred_log)

results['Logistic Regression'] = {
    'best_params': log_clf.best_params_,
    'best_score': log_clf.best_score_,
    'accuracy': accuracy_score(y_test, y_pred_log),
    'precision': precision_score(y_test, y_pred_log),
    'recall': recall_score(y_test, y_pred_log),
    'confusion_matrix': cm_log,
    'predictions': y_pred_log,
    'pred_proba': y_pred_proba_log
}
trained_models['Logistic Regression'] = log_clf.best_estimator_
print("✓ Complete")

# Random Forest
print("[2/5] Training Random Forest...")
rf_parameters = {'n_estimators': [100, 150], 'max_depth': [10, 15]}
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_clf = GridSearchCV(estimator=rf_model, param_grid=rf_parameters, n_jobs=-1, cv=5, scoring='f1')
rf_clf.fit(X, y)
y_pred_rf = rf_clf.predict(X_test)
y_pred_proba_rf = rf_clf.predict_proba(X_test)[:, 1]
cm_rf = confusion_matrix(y_test, y_pred_rf)

results['Random Forest'] = {
    'best_params': rf_clf.best_params_,
    'best_score': rf_clf.best_score_,
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'confusion_matrix': cm_rf,
    'predictions': y_pred_rf,
    'pred_proba': y_pred_proba_rf
}
trained_models['Random Forest'] = rf_clf.best_estimator_
print("✓ Complete")

# XGBoost
print("[3/5] Training XGBoost...")
xgb_parameters = {'n_estimators': [100, 150, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_clf = GridSearchCV(estimator=xgb_model, param_grid=xgb_parameters, n_jobs=-1, cv=5, scoring='f1')
xgb_clf.fit(X, y)
y_pred_xgb = xgb_clf.predict(X_test)
y_pred_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

results['XGBoost'] = {
    'best_params': xgb_clf.best_params_,
    'best_score': xgb_clf.best_score_,
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'precision': precision_score(y_test, y_pred_xgb),
    'recall': recall_score(y_test, y_pred_xgb),
    'confusion_matrix': cm_xgb,
    'predictions': y_pred_xgb,
    'pred_proba': y_pred_proba_xgb
}
trained_models['XGBoost'] = xgb_clf.best_estimator_
print("✓ Complete")

# SVM
print("[4/5] Training SVM...")
svm_parameters = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']}
svm_model = SVC(random_state=42, class_weight='balanced', probability=True)
svm_clf = GridSearchCV(estimator=svm_model, param_grid=svm_parameters, n_jobs=-1, cv=5, scoring='f1')
svm_clf.fit(X, y)
y_pred_svm = svm_clf.predict(X_test)
y_pred_proba_svm = svm_clf.predict_proba(X_test)[:, 1]
cm_svm = confusion_matrix(y_test, y_pred_svm)

results['SVM'] = {
    'best_params': svm_clf.best_params_,
    'best_score': svm_clf.best_score_,
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'precision': precision_score(y_test, y_pred_svm),
    'recall': recall_score(y_test, y_pred_svm),
    'confusion_matrix': cm_svm,
    'predictions': y_pred_svm,
    'pred_proba': y_pred_proba_svm
}
trained_models['SVM'] = svm_clf.best_estimator_
print("✓ Complete")

# Gradient Boosting
print("[5/5] Training Gradient Boosting...")
gbm_parameters = {'n_estimators': [100, 150, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
gbm_model = GradientBoostingClassifier(random_state=42)
gbm_clf = GridSearchCV(estimator=gbm_model, param_grid=gbm_parameters, n_jobs=-1, cv=5, scoring='f1')
gbm_clf.fit(X, y)
y_pred_gbm = gbm_clf.predict(X_test)
y_pred_proba_gbm = gbm_clf.predict_proba(X_test)[:, 1]
cm_gbm = confusion_matrix(y_test, y_pred_gbm)

results['Gradient Boosting'] = {
    'best_params': gbm_clf.best_params_,
    'best_score': gbm_clf.best_score_,
    'accuracy': accuracy_score(y_test, y_pred_gbm),
    'precision': precision_score(y_test, y_pred_gbm),
    'recall': recall_score(y_test, y_pred_gbm),
    'confusion_matrix': cm_gbm,
    'predictions': y_pred_gbm,
    'pred_proba': y_pred_proba_gbm
}
trained_models['Gradient Boosting'] = gbm_clf.best_estimator_
print("✓ Complete")

# =============================================================================
# 8. SAVE TRAINED MODELS
# =============================================================================
print("\n" + "="*80)
print("SAVING TRAINED MODELS")
print("="*80)

for model_name, model in trained_models.items():
    filename = f"model_{model_name.replace(' ', '_').lower()}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"✓ Saved: {filename}")

# Save the scaler as well
with open('scaler.pkl', 'wb') as file:
    pickle.dump(sc, file)
print("✓ Saved: scaler.pkl")

# =============================================================================
# 9. ENHANCED EVALUATION VISUALIZATIONS
# =============================================================================

# 9.1 Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()
model_names = list(results.keys())

for idx, model_name in enumerate(model_names):
    cm = results[model_name]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'],
                cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    axes[idx].set_title(f'{model_name}\nRecall: {results[model_name]["recall"]:.3f}', 
                        fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('Actual', fontsize=11)
    axes[idx].set_xlabel('Predicted', fontsize=11)
axes[5].axis('off')
plt.tight_layout()
plt.savefig('10_all_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.2 Metrics Comparison Bar Chart
summary_data = []
for model_name, metrics in results.items():
    summary_data.append({
        'Model': model_name,
        'Best CV Score': metrics['best_score'],
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'Best Parameters': str(metrics['best_params'])
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Recall', ascending=False)

fig, ax = plt.subplots(figsize=(14, 6))
x = range(len(summary_df))
width = 0.25
bars1 = ax.bar([i - width for i in x], summary_df['Accuracy'], width, 
               label='Accuracy', color='skyblue', edgecolor='black')
bars2 = ax.bar(x, summary_df['Precision'], width, 
               label='Precision', color='lightgreen', edgecolor='black')
bars3 = ax.bar([i + width for i in x], summary_df['Recall'], width, 
               label='Recall', color='salmon', edgecolor='black')

ax.set_xlabel('Models', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(summary_df['Model'], rotation=20, ha='right')
ax.legend()
ax.set_ylim(0.5, 1.0)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('11_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.3 ROC Curves
plt.figure(figsize=(10, 8))
for model_name in model_names:
    y_pred_proba = results[model_name]['pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('12_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.4 Precision-Recall Curves
plt.figure(figsize=(10, 8))
for model_name in model_names:
    y_pred_proba = results[model_name]['pred_proba']
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, label=f'{model_name}', linewidth=2)

plt.xlabel('Recall', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('13_precision_recall_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.5 Model Comparison Radar Chart
from math import pi

categories = ['Accuracy', 'Precision', 'Recall', 'CV Score']
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for idx, model_name in enumerate(model_names):
    values = [
        results[model_name]['accuracy'],
        results[model_name]['precision'],
        results[model_name]['recall'],
        results[model_name]['best_score']
    ]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, fontweight='bold')
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'])
ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)
plt.tight_layout()
plt.savefig('14_performance_radar.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.6 Feature Importance (for tree-based models)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
feature_names = X.columns

# Random Forest
rf_importance = trained_models['Random Forest'].feature_importances_
rf_indices = np.argsort(rf_importance)[-15:]
axes[0].barh(range(len(rf_indices)), rf_importance[rf_indices], color='#3498db')
axes[0].set_yticks(range(len(rf_indices)))
axes[0].set_yticklabels([feature_names[i] for i in rf_indices], fontsize=8)
axes[0].set_xlabel('Importance')
axes[0].set_title('Random Forest - Top 15 Features', fontweight='bold')

# XGBoost
xgb_importance = trained_models['XGBoost'].feature_importances_
xgb_indices = np.argsort(xgb_importance)[-15:]
axes[1].barh(range(len(xgb_indices)), xgb_importance[xgb_indices], color='#2ecc71')
axes[1].set_yticks(range(len(xgb_indices)))
axes[1].set_yticklabels([feature_names[i] for i in xgb_indices], fontsize=8)
axes[1].set_xlabel('Importance')
axes[1].set_title('XGBoost - Top 15 Features', fontweight='bold')

# Gradient Boosting
gbm_importance = trained_models['Gradient Boosting'].feature_importances_
gbm_indices = np.argsort(gbm_importance)[-15:]
axes[2].barh(range(len(gbm_indices)), gbm_importance[gbm_indices], color='#e74c3c')
axes[2].set_yticks(range(len(gbm_indices)))
axes[2].set_yticklabels([feature_names[i] for i in gbm_indices], fontsize=8)
axes[2].set_xlabel('Importance')
axes[2].set_title('Gradient Boosting - Top 15 Features', fontweight='bold')

plt.tight_layout()
plt.savefig('15_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 10. FINAL SUMMARY AND BEST MODEL
# =============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(summary_df.to_string(index=False))

best_model = summary_df.iloc[0]['Model']
print(f"\n{'='*80}")
print(f"BEST MODEL BY RECALL: {best_model}")
print(f"{'='*80}")
print(f"Recall: {summary_df.iloc[0]['Recall']:.4f}")
print(f"Precision: {summary_df.iloc[0]['Precision']:.4f}")
print(f"Accuracy: {summary_df.iloc[0]['Accuracy']:.4f}")
print(f"Best Parameters: {summary_df.iloc[0]['Best Parameters']}")

# Classification Report for Best Model
print(f"\nDetailed Classification Report for {best_model}:")
print("="*80)
y_pred_best = results[best_model]['predictions']
print(classification_report(y_test, y_pred_best, target_names=['No Churn', 'Churn']))

# Save summary to CSV
summary_df.to_csv('model_comparison_summary.csv', index=False)
print("\n✓ Model comparison summary saved to: model_comparison_summary.csv")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Visualizations:")
print("  1. 01_target_distribution.png")
print("  2. 02_binary_features_vs_churn.png")
print("  3. 03_churn_rates_binary.png")
print("  4. 04_internet_service_analysis.png")
print("  5. 05_services_churn_rates.png")
print("  6. 06_contract_payment_analysis.png")
print("  7. 07_numerical_features_analysis.png")
print("  8. 08_correlation_heatmap.png")
print("  9. 09_resampling_comparison.png")
print(" 10. 10_all_confusion_matrices.png")
print(" 11. 11_metrics_comparison.png")
print(" 12. 12_roc_curves.png")
print(" 13. 13_precision_recall_curves.png")
print(" 14. 14_performance_radar.png")
print(" 15. 15_feature_importance.png")

print("\nSaved Models:")
for model_name in trained_models.keys():
    print(f"  - model_{model_name.replace(' ', '_').lower()}.pkl")
print("  - scaler.pkl")

print("\n" + "="*80)