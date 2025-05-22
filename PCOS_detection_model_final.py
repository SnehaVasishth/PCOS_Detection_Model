# %%
import numpy as np 
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    precision_recall_curve, auc, r2_score
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn import tree


# %%
df = pd.read_csv('CLEAN- PCOS SURVEY SPREADSHEET.csv')

# Basic checks
print(df.head())
print(df.isnull().sum())  # No Null values
print(df.info())
print(df.nunique())

# Check unique blood groups
print(df['Can you tell us your blood group ?'].unique())


# %%
df = df.rename(columns={
    "Age (in Years)": "Age",  
    "Weight (in Kg)": "Weight",  
    "Height (in Cm / Feet)": "Height",  
    "Can you tell us your blood group ?": "Blood Group",  
    "After how many months do you get your periods?\n(select 1- if every month/regular)": "Menstrual Cycle Interval",  
    "Have you gained weight recently?": "Recent Weight Gain",  
    "Are you noticing skin darkening recently?": "Skin Darkening",  
    "Do have hair loss/hair thinning/baldness ?": "Hair Loss",  
    "Do you have pimples/acne on your face/jawline ?": "Acne",  
    "Do you eat fast food regularly ?": "Regular Fast Food Consumption",  
    "Do you exercise on a regular basis ?": "Regular Exercise",  
    "Have you been diagnosed with PCOS/PCOD?": "PCOS Diagnosis",  
    "Do you experience mood swings ?": "Mood Swings",  
    "Are your periods regular ?": "Regular Periods",  
    "Do you have excessive body/facial hair growth ?": "Excessive Body/Facial Hair",  
    "How long does your period last ? (in Days)\nexample- 1,2,3,4.....": "Menstrual Duration (Days)"  
})


# %%
print(df['PCOS Diagnosis'].value_counts())
print(df['Regular Periods'].value_counts())


# %%
df.columns

# %%
df['Blood Group'].unique()

# %%
df['Menstrual Duration (Days)'].unique()

# %%
df['Menstrual Duration (Days)'].value_counts()

# %%
df['Menstrual Cycle Interval'].value_counts()

# %%
df['Menstrual Cycle Interval'].unique()

# %%
X = df.drop(columns=['PCOS Diagnosis'])
Y = df['PCOS Diagnosis']
print(X.columns.tolist())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# %%


# %%
def calculate_metrics(model, X_test, Y_test):
    Yhat = model.predict(X_test)
    Y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(Y_test, Y_proba, pos_label=1)

    metrics = pd.DataFrame(index=['Model'], columns=['Acc', 'Prec', 'Rec', 'F1', 'AUC', 'AUPR'])
    metrics.at['Model', 'Acc'] = accuracy_score(Y_test, Yhat)
    metrics.at['Model', 'Prec'] = precision_score(Y_test, Yhat, pos_label=1)
    metrics.at['Model', 'Rec'] = recall_score(Y_test, Yhat, pos_label=1)
    metrics.at['Model', 'F1'] = f1_score(Y_test, Yhat, pos_label=1)
    metrics.at['Model', 'AUC'] = roc_auc_score(Y_test, Y_proba)
    metrics.at['Model', 'AUPR'] = auc(recall, precision)

    return metrics


# %%
rf = RandomForestClassifier().fit(X_train, Y_train)

# Save and Load
with open("pcos_randomforest1.pkl", "wb") as file:
    pickle.dump(rf, file)

with open("pcos_randomforest1.pkl", "rb") as file:
    model = pickle.load(file)

print(calculate_metrics(model, X_test, Y_test))


# %%
dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=10)
dt.fit(X_test, model.predict(X_test))

with open("pcos_decisiontree1.pkl", "wb") as file:
    pickle.dump(dt, file)

with open("pcos_decisiontree1.pkl", "rb") as file:
    model = pickle.load(file)

tree.plot_tree(model, feature_names=X.columns, class_names=['no', 'yes'], filled=True)
print(calculate_metrics(model, X_test, Y_test))


# %%
Yhat = cross_val_predict(rf, X, Y, cv=10)

dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=10).fit(X, Yhat)
Y_hat = dt.predict(X_test)

with open("pcos_stacking_rf_dt.pkl", "wb") as file:
    pickle.dump(dt, file)

with open("pcos_stacking_rf_dt.pkl", "rb") as file:
    model = pickle.load(file)

print(calculate_metrics(model, X_test, Y_test))


# %%
svm = SVC(kernel='linear', probability=True).fit(X_train, Y_train)

with open("pcos_svm1.pkl", "wb") as file:
    pickle.dump(svm, file)

with open("pcos_svm1.pkl", "rb") as file:
    model = pickle.load(file)

print(calculate_metrics(model, X_test, Y_test))


# %%
Yhat = cross_val_predict(svm, X, Y, cv=10)
lr = LogisticRegression(max_iter=1000).fit(X, Yhat)

with open("pcos_stacking_svm_lr.pkl", "wb") as file:
    pickle.dump(lr, file)

with open("pcos_stacking_svm_lr.pkl", "rb") as file:
    model = pickle.load(file)

print(calculate_metrics(model, X_test, Y_test))

coef = pd.DataFrame(model.coef_)
coef.columns = X.columns
print(coef)


# %%
rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, Y_train)

with open("pcos_rf2.pkl", "wb") as file:
    pickle.dump(rf, file)

with open("pcos_rf2.pkl", "rb") as file:
    model = pickle.load(file)

print(calculate_metrics(rf, X_test, Y_test))


# %%
correlation_matrix = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
plt.title("Correlation heatmap of features")
plt.show()


# %%
# Calculate correlation matrix
correlation_matrix = df.corr()

# Get correlation values of all features with 'PCOS Diagnosis'
pcos_corr = correlation_matrix['PCOS Diagnosis'].drop('PCOS Diagnosis')  # remove self-correlation (1.0)

# Sort by correlation value
positive_corr = pcos_corr.sort_values(ascending=False)
negative_corr = pcos_corr.sort_values(ascending=True)

# Display results
print("Highly positively correlated features with PCOS Diagnosis:\n")
print(positive_corr)

print("\nHighly negatively correlated features with PCOS Diagnosis:\n")
print(negative_corr)


# %%
result = permutation_importance(rf, X_test, Y_test, n_repeats=30, random_state=0)
forest_importances = pd.Series(result.importances_mean, index=X.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()


# %%
class_0_importances = {}
class_1_importances = {}

for class_label in [0, 1]:
    X_test_class = X_test[Y_test == class_label]
    Y_test_class = Y_test[Y_test == class_label]

    result = permutation_importance(rf, X_test_class, Y_test_class, n_repeats=30, random_state=0)

    if class_label == 0:
        class_0_importances = result.importances_mean
    else:
        class_1_importances = result.importances_mean

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axs[0].bar(X.columns, class_0_importances, yerr=result.importances_std)
axs[0].set_title("Feature importances for class 0")
axs[0].set_ylabel("Mean accuracy decrease")

axs[1].bar(X.columns, class_1_importances, yerr=result.importances_std)
axs[1].set_title("Feature importances for class 1")
axs[1].set_ylabel("Mean accuracy decrease")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# %%
disp1 = PartialDependenceDisplay.from_estimator(rf, X_test, [4], kind='both', target=1)


# %%
# Load all models
with open("pcos_randomforest1.pkl", "rb") as file: rf1 = pickle.load(file)
with open("pcos_decisiontree1.pkl", "rb") as file: dt1 = pickle.load(file)
with open("pcos_stacking_rf_dt.pkl", "rb") as file: stacking1 = pickle.load(file)
with open("pcos_svm1.pkl", "rb") as file: svm1 = pickle.load(file)
with open("pcos_stacking_svm_lr.pkl", "rb") as file: stacking2 = pickle.load(file)
with open("pcos_rf2.pkl", "rb") as file: rf2 = pickle.load(file)


# %%
# Dictionary of models
models = {
    "Random Forest v1": rf1,
    "Decision Tree": dt1,
    "Stacking RF + DT": stacking1,
    "SVM": svm1,
    "Stacking SVM + LR": stacking2,
    "Random Forest v2": rf2
}

# Evaluate each model and collect results
results_list = []

for name, model in models.items():
    metrics_df = calculate_metrics(model, X_test, Y_test)
    # Flatten the single-row DataFrame to a dict and add model name
    metrics_dict = metrics_df.iloc[0].to_dict()
    metrics_dict["Model"] = name
    results_list.append(metrics_dict)

# Create DataFrame from the list of metric dicts
results_df = pd.DataFrame(results_list)

# Set 'Model' as the index
results_df = results_df.set_index("Model")

# Round for better readability and sort by F1 score
results_df = results_df.round(4).sort_values(by="F1", ascending=False)

# Display the results
print(results_df)


# %%
results_df.style.background_gradient(cmap='Blues').format("{:.2%}")


# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(x=results_df.index, y='F1', data=results_df, palette='viridis')
plt.ylabel("F1 Score")
plt.title("Model Comparison based on F1 Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%


# %% [markdown]
#  Key Features for PCOS Prediction:
# Strong predictors:
# 
# Regular Periods (absence is a major signal)
# 
# Menstrual Cycle Interval
# 
# Moderate contributors:
# 
# Skin Darkening, Weight Gain, Excessive Body/Facial Hair
# 
# Low relevance:
# 
# Blood Group, Acne, Hair Loss, Exercise
# 
# Best Model:
# Stacking SVM + LR gives the highest F1 score 

# %%
test_case_data ={
    
    "Age": 23,
    "Weight": 55,
    "Height": 160,
    "Blood Group": 13,
    "Menstrual Cycle Interval": 1,
    "Recent Weight Gain": 1,
    "Skin Darkening": 0,
    "Hair Loss": 0,
    "Acne": 0,
    "Regular Fast Food Consumption": 1,
    "Regular Exercise": 0,
    "Mood Swings": 1,
    "Regular Periods": 1,
    "Excessive Body/Facial Hair": 0,
    "Menstrual Duration (Days)": 7
}


# Define feature lists (must match the order used during training)
continuous_features = ['Age', 'Weight', 'Height']
discrete_features = ['Blood Group', 'Menstrual Cycle Interval', 'Recent Weight Gain',
                     'Excessive Body/Facial Hair', 'Skin Darkening', 'Hair Loss',
                     'Acne', 'Regular Fast Food Consumption', 'Regular Exercise',
                     'Mood Swings', 'Regular Periods', 'Menstrual Duration (Days)']
all_features = continuous_features + discrete_features

# Convert test case to DataFrame
test_case_df = pd.DataFrame([test_case_data], columns=all_features)

# Ensure best_lr_model is available from running the previous script
# If running separately, load the model and preprocessor here

# Make prediction using the tuned LR model pipeline
try:
    with open("pcos_stacking_svm_lr.pkl", "rb") as file: stacking2 = pickle.load(file)
    predicted_class = stacking2.predict(test_case_df)
    print(f"Predicted Class (0: No PCOS, 1: PCOS): {predicted_class[0]}")
except NameError:
    print("Error: 'best_lr_model' is not found. Ensure the model training script has been run.")
except Exception as e:
    print(f"An error occurred during prediction: {e}")


# %%


# %% [markdown]
# 


