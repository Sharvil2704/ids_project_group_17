import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, precision_recall_curve
from pandas.api.types import CategoricalDtype
import shap

# Step 1: Load the Breast Cancer dataset
url = 'https://archive.ics.uci.edu/static/public/15/data.csv'
df = pd.read_csv(url)

# Step 2: Preprocessing the Dataset
# Handling missing values
imputer = SimpleImputer(strategy='median')
df['Bare_nuclei'] = imputer.fit_transform(df[['Bare_nuclei']])

# Encoding the target variable
label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])

# Save the preprocessed dataset (optional)
df.to_csv('preprocessed_breast_cancer_dataset.csv', index=False)

# Step 3: Preliminary Analysis with Seaborn
# Custom pairplot for selected features
selected_features = ['Clump_thickness', 'Uniformity_of_cell_size', 'Uniformity_of_cell_shape', 
                     'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli', 'Class']

# Pairplot
pairplot = sns.pairplot(df[selected_features], hue='Class', palette='husl', diag_kind='hist')
pairplot.savefig('pairplot_selected_features.png')
plt.show()

# Violin plots for selected features
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
for i, ax in enumerate(axes.flatten()):
    if i < len(selected_features) - 1:
        sns.violinplot(x='Class', y=selected_features[i], data=df, ax=ax, palette='husl')
plt.tight_layout()
plt.savefig('violin_plots_selected_features.png')
plt.show()

# Bar plots for categorical features
categorical_features = ['Bare_nuclei']
for feature in categorical_features:
    plt.figure(figsize=(12, 6))
    df[feature] = df[feature].astype(str)  # Convert to string
    plt.xticks(rotation=90)
    sns.barplot(x='Class', y=feature, data=df, palette='husl')
    plt.title(f'Bar Plot for {feature}')
    plt.xlabel('Class')
    plt.ylabel(feature)
    plt.savefig(f'bar_plot_{feature}.png')
    plt.show()

# Count plot for the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df, palette='husl')
plt.title('Count Plot for Target Variable (Class)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('count_plot_target_variable.png')
plt.show()

# Correlation heatmap
correlation_matrix = df[selected_features].corr()
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(correlation_matrix, cmap='coolwarm')
fig.colorbar(cax)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.savefig('correlation_heatmap.png')
plt.show()

# Correlation matrix table
correlation_table = df[selected_features].corr()
print("Correlation Matrix Table:")
print(correlation_table)

# Step 4: Machine Learning Classification
X = df.drop(['Sample_code_number', 'Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Feature importances
feature_importances = model.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, feature_importances))

# Sort features by importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x=[feature[0] for feature in sorted_features], y=[feature[1] for feature in sorted_features], palette='husl')
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importances')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.savefig('feature_importances.png')
plt.show()

# Step 5: Predict probability of breast cancer and display along with ID
df['Probability_of_breast_cancer'] = model.predict_proba(X)[:, 1]

# Display ID and Probability
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Sample_code_number', y='Probability_of_breast_cancer', hue='Class', data=df, palette='husl')
plt.title('Probability of Breast Cancer Prediction')
plt.xlabel('Sample Code Number')
plt.ylabel('Probability of Breast Cancer')
plt.savefig('probability_of_breast_cancer_prediction.png')
plt.show()


 
# ROC Curve and AUC Score
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.scatter(fpr, tpr, c='red', marker='x', label='Highlighted Point')  # Highlight a specific point
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
y_pred_binary = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_binary)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Distribution Plots for important features
important_features = [feature[0] for feature in sorted_features[:3]]  # Consider top 3 features
for feature in important_features:
    plt.figure(figsize=(10, 6))
    
    # Check if the plot should be created based on the nature of the variable
    if pd.api.types.is_categorical_dtype(df[feature]):
        if not df[feature].nunique() > 1:
            print(f"Skipping {feature} plot as it has only one unique value.")
            continue
        
        sns.countplot(x=feature, hue='Class', data=df, palette='husl')
        plt.title(f'Count of {feature} for Benign and Malignant Tumors')
        plt.xlabel(feature)
        plt.ylabel('Count')
    else:
        if df[feature].nunique() <= 1:
            print(f"Skipping {feature} plot as it has only one unique value.")
            continue
        
        sns.kdeplot(data=df, x=feature, hue='Class', fill=True, common_norm=False, palette='husl')
        plt.title(f'Distribution of {feature} for Benign and Malignant Tumors')
        plt.xlabel(feature)
        plt.ylabel('Density')

    plt.show()
    






# Step 6: SHAP (Optional - Requires shap library)
# Interpret the model's predictions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, plot_type='bar', show=False)
plt.show()


