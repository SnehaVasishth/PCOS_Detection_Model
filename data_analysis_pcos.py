# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, cross_val_predict # cross_val_predict needed for old stacking evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc # Import necessary metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier # Needed for model types
from sklearn.tree import DecisionTreeClassifier # Needed for model type
from sklearn.svm import SVC # Needed for model type
from sklearn.linear_model import LogisticRegression # Needed for model type
from sklearn.neighbors import KNeighborsClassifier # Needed for model type
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Needed for pipelines
from sklearn.compose import ColumnTransformer # Needed for pipelines
from sklearn.pipeline import Pipeline # Needed for pipelines
from imblearn.pipeline import Pipeline as ImbPipeline # Needed for SMOTE pipelines
from imblearn.over_sampling import SMOTE # Needed for SMOTE
# import xgboost as xgb # Needed for XGBoost
import warnings

# %%
df=pd.read_csv('CLEAN- PCOS SURVEY SPREADSHEET.csv')

# %%
df.columns

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
df.columns

# %%
df.dtypes

# %%
dis_col=[]
cont_col=[]
for col in df.columns:
    if df[col].dtypes in ['int64','float64']:
        if df[col].nunique() > 15:
            cont_col.append(col)
        else:
            dis_col.append(col)
        

# %%
print("Discrete Columns: ",dis_col)
print("Continuous Columns: ",cont_col)

# %%
dis_features_col=dis_col
dis_features_col.remove('PCOS Diagnosis')

# %%
for col in dis_features_col:
    df[col]=df[col].astype('category')

# %%
df.isnull().sum()

# %%
df.nunique()

# %%
df.duplicated().sum()

# %%
df=df.drop_duplicates()

# %%
df.duplicated().sum()

# %%
df.shape

# %%
print('discrete features:',dis_features_col)
print('continuous features:',cont_col)
print('target feature:', 'PCOS Diagnosis')

# %%
for col in dis_features_col:
    print(df[col].value_counts())
    sns.countplot(x=col,data=df,hue='PCOS Diagnosis')
    plt.title(col)
    plt.show()


sns.countplot(x='PCOS Diagnosis',data=df)
plt.show()

# %%
for col in dis_features_col:
    pcos_rate=df[df['PCOS Diagnosis']==1][col].groupby(df[col]).size()/df.groupby(df[col]).size()
    pcos_rate.plot(kind='bar')
    plt.title(f'PCOS Rate by {col}')
    plt.show()


    prop_df=df.groupby(col)['PCOS Diagnosis'].value_counts(normalize=True).rename('Prop').reset_index()

    sns.barplot(x=col,y='Prop',hue='PCOS Diagnosis',data=prop_df)
    plt.title(f'PCOS Proportion by {col}')
    plt.show()

# %%
#chi-squaretest

import scipy.stats as stats
from scipy.stats import chi2_contingency

associated_features={}
for col in dis_features_col:
    contigency=pd.crosstab(df[col],df['PCOS Diagnosis'])
    chi2,p,dof,expected=chi2_contingency(contigency)
    print(f"{col}- p-value:{p}")

    if p<0.05:
        print(f"{col} is associated with PCOS")
        associated_features[col]=p

    else:
        print(f"{col} is not associated with PCOS")



# %% [markdown]
# ##Features which are correlated with PCOS through chi-square test
# 
# 

# %%
for index,value in associated_features.items():
    print(f"{index} is associated with PCOS with p-value {value}")

# %%
for col in cont_col:
    print(df[col].describe())
    print(f"Median of {col} is {df[col].median()}")
    
    print(f"Skewness of {col} is {df[col].skew()}")
    print(f"Kurtosis of {col} is {df[col].kurt()}")
    sns.histplot(df[col],kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    sns.boxplot(x=col,data=df)
    plt.title(f"Box plot of {col}")
    plt.show()



# %%
len(df[df['Height']< 140])

# %%
Q1=df['Height'].quantile(0.25)
Q2=df['Height'].quantile(0.75)

IQR=Q2-Q1
lower_bound=Q1-1.5*IQR

outliers=df[df['Height']<lower_bound]['Height']
print(outliers)

median_height=df['Height'].median()

df['Height']=df['Height'].apply(lambda x:median_height if x<lower_bound else x)

sns.histplot(df['Height'],kde=True)
plt.show()

sns.boxplot(x='Height',data=df)
plt.show()


# %% [markdown]
# 1. Age is Right Skewed
# 2. Weight is Normally distributed
# 3. Height had some extreme outliers which are removed by replacing with its median value

# %%
from scipy.stats import shapiro
normal_col=[]

for col in cont_col:
    stat,p=shapiro(df[col])
    print(f"{col}- p value:{p}")

    if p>0.05:
        print(f"{col} is normally distributed")
        normal_col.append(col)

    else:
        print(f"{col} is not normally distributed")

# %%
for col in cont_col:

    plt.figure(figsize=(6,4))
    sns.boxplot(data=df,x='PCOS Diagnosis',y=col)
    plt.title(f'{col} vs PCOS')
    plt.show()

# %%
#through boxplot of weight it can be oserved that average weight of PCOS patient is higher than non-PCOS patients

# %%
from scipy.stats import ttest_ind

for col in cont_col:
    pcos=df[df['PCOS Diagnosis']==1][col]
    nonpcos=df[df['PCOS Diagnosis']==0][col]

    stat,p=ttest_ind(pcos,nonpcos,equal_var=False)
    print(f"{col}- p value:{p}")
    if p<0.05:
        print(f"{col} is associated with PCOS")
    else:
        print(f"{col} is not associated with PCOS")

# %%
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %%
corr_matrix= df.corr(method='pearson')  
target_corr=corr_matrix['PCOS Diagnosis'].drop('PCOS Diagnosis')   
sorted_corr=target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)
print(f"\nFeatures sorted by absolute correlation with PCOS Diagnosis (Descending):")
print(sorted_corr)

# %%


sns.pairplot(df,hue='PCOS Diagnosis')
plt.show()

# %%
# from above pairplot we can see that PCOS patients have higher weights

# %%
#discrete vs discrete feature

# %%
import itertools
dis_pair= list(itertools.combinations(dis_features_col, 2))
significant_dis_assoc={}

for col1,col2 in dis_pair:
    print(f"{col1} vs {col2}")

    temp_df=df[[col1,col2]].dropna()
    crosstab_df=pd.crosstab(temp_df[col1],temp_df[col2])

    if(min(crosstab_df.shape)>1 and crosstab_df.sum().sum()>0):

        try:
            chi2,p,dof,expected=chi2_contingency(crosstab_df)
            print(f"Chi-Squared Test Result (p-value): {p:.4f}")
            if p<0.05:
                print(f"Interpretation: There is a statistically significant association between {col1} and {col2} (p < 0.05).")
                if col1 not in significant_dis_assoc:
                    significant_dis_assoc[col1]={}
                significant_dis_assoc[col1][col2]=p
            else:
                print(f"Interpretation: No statistically significant association between {col1} and {col2} (p >= 0.05).")
        except ValueError as e:
            print(f"Error in Chi-Squared Test: {e}")
    else:
        print(f"Error: Not enough data to perform Chi-Squared Test for {col1} and {col2}.")






# %%
for index,value in significant_dis_assoc.items():
    print(f"{index} is associated with PCOS with p-value {value}")

# %%


# %%
##ML Modeliing

# %%


# %%
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
import shap
import warnings

# %%
X=df.drop('PCOS Diagnosis',axis=1)
y=df['PCOS Diagnosis']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print(f"\nTrain features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Train target shape: {y_train.shape}")
print(f"Test target shape: {y_test.shape}")

# %%
continuous_transformer=Pipeline(steps=[
    ('scaler',StandardScaler())
])

discrete_transformer=Pipeline(steps=[
    ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))   


])

preprocessor=ColumnTransformer(
    transformers=[
        ('num',continuous_transformer,cont_col),
        ('cat',discrete_transformer,dis_features_col)
    ]
)


# %%
df['PCOS Diagnosis'].dtypes

# %%
X_train_trans=preprocessor.fit_transform(X_train)
X_test_trans=preprocessor.transform(X_test)

# %%
onehot_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(dis_features_col)
    # Combine with continuous feature names
processed_feature_names = list(cont_col) + list(onehot_feature_names)
print(f"\nExample of processed feature names (first 10): {processed_feature_names[:10]}...")
print(f"Total processed features: {len(processed_feature_names)}")

# %%
models={
    'Logistic Regression':  LogisticRegression(solver='liblinear',class_weight='balanced',random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced',random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True,class_weight='balanced',random_state=42),
    'KNN': KNeighborsClassifier(),
}

# %%
scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

results={}

for name,model in models.items():
    print(f"\nTraining and evaluating {name}...")
    try:
        cv_results = cross_validate(model, X_train_trans, y_train, cv=5, scoring=scoring, n_jobs=-1)
        results[name] = {metric: cv_results[f'test_{metric}'].mean() for metric in scoring}
        print(f"{name} Cross-Validation Results (Mean across folds):")
        for metric, value in results[name].items():
            print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error during cross-validation for {name}: {e}")
        results[name] = {metric: np.nan for metric in scoring}

        



# %%
print("\n--- Cross-Validation Summary (Mean Scores) ---")
cv_summary_df = pd.DataFrame(results).T
print(cv_summary_df.round(4))

print("\nPotential best models based on Recall:")
print(cv_summary_df.sort_values(by='recall', ascending=False)['recall'].head())
print("\nPotential best models based on F1-Score:")
print(cv_summary_df.sort_values(by='f1', ascending=False)['f1'].head())
print("\nPotential best models based on ROC AUC:")
print(cv_summary_df.sort_values(by='roc_auc', ascending=False)['roc_auc'].head())

# %%
param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

pipeline_rf_tuned = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))])

grid_search_rf = GridSearchCV(pipeline_rf_tuned, param_grid_rf, cv=5, scoring='recall', n_jobs=-1)

print("Performing GridSearchCV for Random Forest...")
grid_search_rf.fit(X_train, y_train)

print("\nBest parameters found for Random Forest: ", grid_search_rf.best_params_)
print("Best cross-validation score (Recall) for Random Forest: ", grid_search_rf.best_score_)

best_rf_model = grid_search_rf.best_estimator_
print("\nBest tuned Random Forest model obtained.")

# %%
param_grid_lr = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

pipeline_lr_tuned = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42))])

grid_search_lr = GridSearchCV(pipeline_lr_tuned, param_grid_lr, cv=5, scoring='recall', n_jobs=-1)

print("Performing GridSearchCV for Logistic Regression...")
grid_search_lr.fit(X_train, y_train)

print("\nBest parameters found for Logistic Regression: ", grid_search_lr.best_params_)
print("Best cross-validation score (Recall) for Logistic Regression: ", grid_search_lr.best_score_)

best_lr_model = grid_search_lr.best_estimator_
print("\nBest tuned Logistic Regression model obtained.")

# %%
param_grid_svm = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__kernel': ['rbf', 'linear'] # 'linear' kernel doesn't use gamma
}

pipeline_svm_tuned = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', SVC(probability=True, class_weight='balanced', random_state=42))])

grid_search_svm = GridSearchCV(pipeline_svm_tuned, param_grid_svm, cv=5, scoring='recall', n_jobs=-1)

print("Performing GridSearchCV for SVM...")
grid_search_svm.fit(X_train, y_train)

print("\nBest parameters found for SVM: ", grid_search_svm.best_params_)
print("Best cross-validation score (Recall) for SVM: ", grid_search_svm.best_score_)

best_svm_model = grid_search_svm.best_estimator_
print("\nBest tuned SVM model obtained.")

# %%
param_grid_gb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5],
    'classifier__subsample': [0.8, 1.0],
    'classifier__max_features': ['sqrt', 'log2', None]
}

# Pipeline includes preprocessing and the classifier (no SMOTE)
pipeline_gb_tuned = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', GradientBoostingClassifier(random_state=42))])

grid_search_gb = GridSearchCV(pipeline_gb_tuned, param_grid_gb, cv=5, scoring='recall', n_jobs=-1)

print("Performing GridSearchCV for Gradient Boosting...")
# Fit GridSearchCV on the ORIGINAL X_train DataFrame
grid_search_gb.fit(X_train, y_train)

print("\nBest parameters found for Gradient Boosting: ", grid_search_gb.best_params_)
print("Best cross-validation score (Recall) for Gradient Boosting: ", grid_search_gb.best_score_)

best_gb_model = grid_search_gb.best_estimator_
print("\nBest tuned Gradient Boosting model obtained.")

# %%
print("\n--- Tuned Random Forest Evaluation ---")
y_pred_rf = best_rf_model.predict(X_test)
y_proba_rf = best_rf_model.predict_proba(X_test)[:, 1]
print("\nClassification Report (Tuned Random Forest):")
print(classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix (Tuned Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
print(f"\nROC AUC Score (Tuned Random Forest): {roc_auc_rf:.4f}")

# %%
print("\n--- Tuned Logistic Regression Evaluation ---")
y_pred_lr = best_lr_model.predict(X_test)
y_proba_lr = best_lr_model.predict_proba(X_test)[:, 1]
print("\nClassification Report (Tuned Logistic Regression):")
print(classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix (Tuned Logistic Regression):")
print(confusion_matrix(y_test, y_pred_lr))
roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
print(f"\nROC AUC Score (Tuned Logistic Regression): {roc_auc_lr:.4f}")

# %%
print("\n--- Tuned SVM Evaluation ---")
y_pred_svm = best_svm_model.predict(X_test)
y_proba_svm = best_svm_model.predict_proba(X_test)[:, 1]
print("\nClassification Report (Tuned SVM):")
print(classification_report(y_test, y_pred_svm))
print("\nConfusion Matrix (Tuned SVM):")
print(confusion_matrix(y_test, y_pred_svm))
roc_auc_svm = roc_auc_score(y_test, y_proba_svm)
print(f"\nROC AUC Score (Tuned SVM): {roc_auc_svm:.4f}")


# %%
print("\n--- Tuned Gradient Boosting Evaluation (without SMOTE) ---")
y_pred_gb = best_gb_model.predict(X_test)
y_proba_gb = best_gb_model.predict_proba(X_test)[:, 1]
print("\nClassification Report (Tuned Gradient Boosting without SMOTE):")
print(classification_report(y_test, y_pred_gb))
print("\nConfusion Matrix (Tuned Gradient Boosting without SMOTE):")
print(confusion_matrix(y_test, y_pred_gb))
roc_auc_gb = roc_auc_score(y_test, y_proba_gb)
print(f"\nROC AUC Score (Tuned Gradient Boosting without SMOTE): {roc_auc_gb:.4f}")

# %%


# %%
smote_df=pd.read_csv('CLEAN- PCOS SURVEY SPREADSHEET.csv')

# %%
smote_df = smote_df.rename(columns={
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
from imblearn.over_sampling import SMOTE # Import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Use imblearn's pipeline for integrating SMOTE
import shap

# %%
smote_X=smote_df.drop('PCOS Diagnosis',axis=1)
smote_y=smote_df['PCOS Diagnosis']

# %%
smote_X_train, smote_X_test, smote_y_train, smote_y_test = train_test_split(smote_X, smote_y, test_size=0.25, random_state=42, stratify=smote_y)

print(f"\nTrain features shape: {smote_X_train.shape}")
print(f"Test features shape: {smote_X_test.shape}")
print(f"Train target shape: {smote_y_train.shape}")
print(f"Test target shape: {smote_y_test.shape}")

# %%
smote_continuous_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

smote_discrete_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

smote_preprocessor = ColumnTransformer(
    transformers=[
        ('num', smote_continuous_transformer, cont_col),
        ('cat', smote_discrete_transformer, dis_features_col)
    ],
    remainder='passthrough'
)

# %%
temp_processed_X_train = smote_preprocessor.fit_transform(smote_X_train)
onehot_feature_names_smote = smote_preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(dis_features_col)
smote_processed_feature_names = list(cont_col) + list(onehot_feature_names_smote)
print(f"\nSuccessfully retrieved processed feature names for SMOTE workflow. Total: {len(smote_processed_feature_names)}")


# %%
smote_models = {
    'Logistic Regression (SMOTE)': LogisticRegression(solver='liblinear', random_state=42),
    'Random Forest (SMOTE)': RandomForestClassifier(random_state=42),
    'Gradient Boosting (SMOTE)': GradientBoostingClassifier(random_state=42),
    'SVM (SMOTE)': SVC(probability=True, random_state=42),
    'KNN (SMOTE)': KNeighborsClassifier()
}

# Define scoring metrics for cross-validation
smote_scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

smote_results = {}

# %%
for name, model in smote_models.items():
    print(f"\nTraining and evaluating {name}...")

    # Create a pipeline that includes preprocessing, SMOTE, and the model
    smote_pipeline = ImbPipeline(steps=[('preprocessor', smote_preprocessor),
                                         ('smote', SMOTE(random_state=42)), # SMOTE step
                                         ('classifier', model)]) # Model step

    try:
        # Use cross_validate on the ORIGINAL X_train, y_train
        # The pipeline handles preprocessing and SMOTE within each fold
        smote_cv_results = cross_validate(smote_pipeline, smote_X_train, smote_y_train, cv=5, scoring=smote_scoring, n_jobs=-1)
        smote_results[name] = {metric: smote_cv_results[f'test_{metric}'].mean() for metric in smote_scoring}
        print(f"{name} Cross-Validation Results (Mean across folds):")
        for metric, value in smote_results[name].items():
            print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error during cross-validation for {name}: {e}")
        smote_results[name] = {metric: np.nan for metric in smote_scoring}


# --- Summarize Cross-Validation Results with SMOTE ---
print("\n--- Cross-Validation Summary (Mean Scores with SMOTE) ---")
smote_cv_summary_df = pd.DataFrame(smote_results).T
print(smote_cv_summary_df.round(4))

print("\nPotential best models with SMOTE based on Recall:")
print(smote_cv_summary_df.sort_values(by='recall', ascending=False)['recall'].head())
print("\nPotential best models with SMOTE based on F1-Score:")
print(smote_cv_summary_df.sort_values(by='f1', ascending=False)['f1'].head())
print("\nPotential best models with SMOTE based on ROC AUC:")
print(smote_cv_summary_df.sort_values(by='roc_auc', ascending=False)['roc_auc'].head())


# %%
smote_param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Pipeline includes preprocessing, SMOTE, and the classifier
smote_pipeline_rf_tuned = ImbPipeline(steps=[('preprocessor', smote_preprocessor),
                                             ('smote', SMOTE(random_state=42)),
                                             ('classifier', RandomForestClassifier(random_state=42))])

smote_grid_search_rf = GridSearchCV(smote_pipeline_rf_tuned, smote_param_grid_rf, cv=5, scoring='recall', n_jobs=-1)

print("Performing GridSearchCV for Random Forest with SMOTE...")
# Fit GridSearchCV on the ORIGINAL smote_X_train DataFrame
smote_grid_search_rf.fit(smote_X_train, smote_y_train)

print("\nBest parameters found for Random Forest with SMOTE: ", smote_grid_search_rf.best_params_)
print("Best cross-validation score (Recall) for Random Forest with SMOTE: ", smote_grid_search_rf.best_score_)

best_smote_rf_model = smote_grid_search_rf.best_estimator_
print("\nBest tuned Random Forest model with SMOTE obtained.")

# %%
smote_param_grid_lr = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

smote_pipeline_lr_tuned = ImbPipeline(steps=[('preprocessor', smote_preprocessor),
                                             ('smote', SMOTE(random_state=42)),
                                             ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

smote_grid_search_lr = GridSearchCV(smote_pipeline_lr_tuned, smote_param_grid_lr, cv=5, scoring='recall', n_jobs=-1)

print("Performing GridSearchCV for Logistic Regression with SMOTE...")
# Fit GridSearchCV on the ORIGINAL smote_X_train DataFrame
smote_grid_search_lr.fit(smote_X_train, smote_y_train)

print("\nBest parameters found for Logistic Regression with SMOTE: ", smote_grid_search_lr.best_params_)
print("Best cross-validation score (Recall) for Logistic Regression with SMOTE: ", smote_grid_search_lr.best_score_)

best_smote_lr_model = smote_grid_search_lr.best_estimator_
print("\nBest tuned Logistic Regression model with SMOTE obtained.")


# %%
smote_param_grid_svm = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__kernel': ['rbf', 'linear']
}

smote_pipeline_svm_tuned = ImbPipeline(steps=[('preprocessor', smote_preprocessor),
                                              ('smote', SMOTE(random_state=42)),
                                              ('classifier', SVC(probability=True, random_state=42))])

smote_grid_search_svm = GridSearchCV(smote_pipeline_svm_tuned, smote_param_grid_svm, cv=5, scoring='recall', n_jobs=-1)

print("Performing GridSearchCV for SVM with SMOTE...")
# Fit GridSearchCV on the ORIGINAL smote_X_train DataFrame
smote_grid_search_svm.fit(smote_X_train, smote_y_train)

print("\nBest parameters found for SVM with SMOTE: ", smote_grid_search_svm.best_params_)
print("Best cross-validation score (Recall) for SVM with SMOTE: ", smote_grid_search_svm.best_score_)

best_smote_svm_model = smote_grid_search_svm.best_estimator_
print("\nBest tuned SVM model with SMOTE obtained.")


# %%
y_pred_smote_rf = best_smote_rf_model.predict(smote_X_test)
y_proba_smote_rf = best_smote_rf_model.predict_proba(smote_X_test)[:, 1]
print("\nClassification Report (Tuned Random Forest with SMOTE):")
print(classification_report(smote_y_test, y_pred_smote_rf))
print("\nConfusion Matrix (Tuned Random Forest with SMOTE):")
print(confusion_matrix(smote_y_test, y_pred_smote_rf))
roc_auc_smote_rf = roc_auc_score(smote_y_test, y_proba_smote_rf)
print(f"\nROC AUC Score (Tuned Random Forest with SMOTE): {roc_auc_smote_rf:.4f}")


# %%
y_pred_smote_lr = best_smote_lr_model.predict(smote_X_test)
y_proba_smote_lr = best_smote_lr_model.predict_proba(smote_X_test)[:, 1]
print("\nClassification Report (Tuned Logistic Regression with SMOTE):")
print(classification_report(smote_y_test, y_pred_smote_lr))
print("\nConfusion Matrix (Tuned Logistic Regression with SMOTE):")
print(confusion_matrix(smote_y_test, y_pred_smote_lr))
roc_auc_smote_lr = roc_auc_score(smote_y_test, y_proba_smote_lr)
print(f"\nROC AUC Score (Tuned Logistic Regression with SMOTE): {roc_auc_smote_lr:.4f}")


# %%
y_pred_smote_svm = best_smote_svm_model.predict(smote_X_test)
y_proba_smote_svm = best_smote_svm_model.predict_proba(smote_X_test)[:, 1]
print("\nClassification Report (Tuned SVM with SMOTE):")
print(classification_report(smote_y_test, y_pred_smote_svm))
print("\nConfusion Matrix (Tuned SVM with SMOTE):")
print(confusion_matrix(smote_y_test, y_pred_smote_svm))
roc_auc_smote_svm = roc_auc_score(smote_y_test, y_proba_smote_svm)
print(f"\nROC AUC Score (Tuned SVM with SMOTE): {roc_auc_smote_svm:.4f}")

# %%
smote_param_grid_gb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5],
    'classifier__subsample': [0.8, 1.0],
    'classifier__max_features': ['sqrt', 'log2', None]
}

# Pipeline includes preprocessing, SMOTE, and the classifier
smote_pipeline_gb_tuned = ImbPipeline(steps=[('preprocessor', smote_preprocessor),
                                             ('smote', SMOTE(random_state=42)),
                                             ('classifier', GradientBoostingClassifier(random_state=42))])

smote_grid_search_gb = GridSearchCV(smote_pipeline_gb_tuned, smote_param_grid_gb, cv=5, scoring='recall', n_jobs=-1)

print("Performing GridSearchCV for Gradient Boosting with SMOTE...")
# Fit GridSearchCV on the ORIGINAL smote_X_train DataFrame
smote_grid_search_gb.fit(smote_X_train, smote_y_train)

print("\nBest parameters found for Gradient Boosting with SMOTE: ", smote_grid_search_gb.best_params_)
print("Best cross-validation score (Recall) for Gradient Boosting with SMOTE: ", smote_grid_search_gb.best_score_)

best_smote_gb_model = smote_grid_search_gb.best_estimator_
print("\nBest tuned Gradient Boosting model with SMOTE obtained.")

# %%
y_pred_smote_gb = best_smote_gb_model.predict(smote_X_test)
y_proba_smote_gb = best_smote_gb_model.predict_proba(smote_X_test)[:, 1]
print("\nClassification Report (Tuned Gradient Boosting with SMOTE):")
print(classification_report(smote_y_test, y_pred_smote_gb))
print("\nConfusion Matrix (Tuned Gradient Boosting with SMOTE):")
print(confusion_matrix(smote_y_test, y_pred_smote_gb))
roc_auc_smote_gb = roc_auc_score(smote_y_test, y_proba_smote_gb)
print(f"\nROC AUC Score (Tuned Gradient Boosting with SMOTE): {roc_auc_smote_gb:.4f}")

# %%
if smote_processed_feature_names and hasattr(best_smote_lr_model.named_steps['classifier'], 'coef_'):
    smote_lr_coefficients = pd.Series(best_smote_lr_model.named_steps['classifier'].coef_[0], index=smote_processed_feature_names)
    sorted_smote_lr_coefficients = smote_lr_coefficients.reindex(smote_lr_coefficients.abs().sort_values(ascending=False).index)
    print(sorted_smote_lr_coefficients.round(4))

    print("\nInterpretation of LR Coefficients (SMOTE):")
    print("- Coefficients represent the change in log-odds of PCOS per unit/std dev change on the SMOTEd data.")
    print("- Odds Ratios (exp(coef)) indicate the multiplicative change in odds.")
    print("  - Interpretation is similar to before, but reflects the model learned on the balanced dataset.")


    print("\nTop 10 Features by Odds Ratio (Magnitude) from Tuned LR Model with SMOTE:")
    smote_lr_odds_ratios = np.exp(smote_lr_coefficients)
    sorted_smote_lr_odds_ratios = smote_lr_odds_ratios.reindex(smote_lr_odds_ratios.abs().sort_values(ascending=False).index)
    print(sorted_smote_lr_odds_ratios.head(10).round(4))

else:
    print("Processed feature names or Tuned LR coefficients with SMOTE not available.")


# %%
if smote_processed_feature_names and hasattr(best_smote_rf_model.named_steps['classifier'], 'feature_importances_'):
    smote_rf_importances = pd.Series(best_smote_rf_model.named_steps['classifier'].feature_importances_, index=smote_processed_feature_names)
    sorted_smote_rf_importances = smote_rf_importances.sort_values(ascending=False)
    print(sorted_smote_rf_importances.round(4))

    print("\nInterpretation of RF Feature Importance (SMOTE):")
    print("- Importance scores reflect feature utility for splitting nodes on the SMOTEd data.")


else:
    print("Random Forest feature importances with SMOTE not available.")


# %%
print("\nGradient Boosting Feature Importance (from Tuned GB Model with SMOTE - Mean Decrease in Impurity):")
# Gradient Boosting also has feature_importances_
if smote_processed_feature_names and hasattr(best_smote_gb_model.named_steps['classifier'], 'feature_importances_'):
    smote_gb_importances = pd.Series(best_smote_gb_model.named_steps['classifier'].feature_importances_, index=smote_processed_feature_names)
    sorted_smote_gb_importances = smote_gb_importances.sort_values(ascending=False)
    print(sorted_smote_gb_importances.round(4))

    print("\nInterpretation of GB Feature Importance (SMOTE):")
    print("- Similar to Random Forest, this score is based on impurity reduction across the boosting stages on the SMOTEd data.")


else:
    print("Gradient Boosting feature importances with SMOTE not available.")

# %%
print("\nGenerating SHAP Summary Plot for Tuned Random Forest with SMOTE...")

try:
    # SHAP needs the data that was fed into the classifier step of the pipeline
    # We need to transform X_train using the preprocessor from the best_smote_rf_model pipeline
    explainer_smote = shap.TreeExplainer(best_smote_rf_model.named_steps['classifier'])
    # Access the preprocessor from the tuned pipeline to ensure consistent transformation
    X_train_transformed_for_shap_smote = best_smote_rf_model.named_steps['preprocessor'].transform(smote_X_train)

    # Calculate SHAP values for a sample of the training data (can be slow on large datasets)
    # Using a smaller sample for faster computation
    X_sample_for_shap = X_train_transformed_for_shap_smote[:100] # Take the data sample first

    # --- Debugging Prints ---
    print(f"Shape of X_sample_for_shap: {np.shape(X_sample_for_shap)}")
    print(f"Type of X_sample_for_shap: {type(X_sample_for_shap)}")
    # --- End Debugging Prints ---


    shap_values_smote = explainer_smote.shap_values(X_sample_for_shap) # Calculate SHAP for the sample

    # --- Debugging Prints ---
    print(f"Shape of shap_values_smote: {np.shape(shap_values_smote)}")
    print(f"Type of shap_values_smote: {type(shap_values_smote)}")
    # --- End Debugging Prints ---


    # Get the SHAP values for the positive class
    # For multi-output models (like multi-class classification), shap_values returns a list.
    # For binary classification, it might return a list of two arrays (one per class)
    # or sometimes a single array depending on the explainer/version.
    if isinstance(shap_values_smote, list) and len(shap_values_smote) > 1:
        shap_values_positive_class = shap_values_smote[1] # SHAP values for the positive class (index 1)
        print(f"Shape of shap_values_positive_class (from list): {np.shape(shap_values_positive_class)}") # Updated print
    elif isinstance(shap_values_smote, np.ndarray) and shap_values_smote.ndim == 3: # Check if it's a 3D array
         # If it's a 3D array (samples, features, classes), slice for the positive class
         shap_values_positive_class = shap_values_smote[:, :, 1] # Take the slice for the positive class
         print(f"Shape of shap_values_positive_class (from 3D array slice): {np.shape(shap_values_positive_class)}") # Updated print
    elif isinstance(shap_values_smote, np.ndarray) and shap_values_smote.ndim == 2:
         # If it's a 2D array, assume it's for the positive class (common for some binary explainers)
         shap_values_positive_class = shap_values_smote
         print(f"Shape of shap_values_positive_class (from 2D array): {np.shape(shap_values_positive_class)}") # Updated print
    else:
         print("Unexpected type or shape for shap_values_smote")
         raise TypeError("Unexpected SHAP values format")


    if smote_processed_feature_names:
         # Ensure the number of feature names matches the number of features in the SHAP values
         if len(smote_processed_feature_names) == shap_values_positive_class.shape[1]:
             shap_feature_names_smote = smote_processed_feature_names
         else:
             print(f"Warning: Number of processed feature names ({len(smote_processed_feature_names)}) does not match SHAP values feature dimension ({shap_values_positive_class.shape[1]}). Using generic names.")
             shap_feature_names_smote = [f'feature_{i}' for i in range(X_train_transformed_for_shap_smote.shape[1])]

    else:
         shap_feature_names_smote = [f'feature_{i}' for i in range(X_train_transformed_for_shap_smote.shape[1])]
         print("Using generic feature names for SHAP plot as specific names were not available.")

    # SHAP summary plot (shows overall importance and impact direction)
    # Pass the SHAP values for the positive class and the corresponding data sample
    shap.summary_plot(shap_values_positive_class, X_sample_for_shap, feature_names=shap_feature_names_smote)
    plt.suptitle('SHAP Summary Plot (SMOTE - Impact on PCOS Diagnosis=1)', y=1.02)
    plt.show()

except Exception as e:
    print(f"\nError generating SHAP plot with SMOTE: {e}")
    print("SHAP might not be compatible with all models or require specific configurations.")


# %%
if smote_processed_feature_names and 'best_smote_lr_model' in locals() and 'shap_values_smote' in locals() and 'best_smote_rf_model' in locals() and 'best_smote_gb_model' in locals() and hasattr(best_smote_lr_model.named_steps['classifier'], 'coef_') and hasattr(best_smote_rf_model.named_steps['classifier'], 'feature_importances_') and hasattr(best_smote_gb_model.named_steps['classifier'], 'feature_importances_'):

    # Recalculate LR coefficients and Odds Ratios from the best tuned LR model with SMOTE
    smote_lr_coefficients_tuned = pd.Series(best_smote_lr_model.named_steps['classifier'].coef_[0], index=smote_processed_feature_names)
    smote_lr_odds_ratios_tuned = np.exp(smote_lr_coefficients_tuned)

    # Get RF importances from the best tuned RF model with SMOTE
    smote_rf_importances_tuned = pd.Series(best_smote_rf_model.named_steps['classifier'].feature_importances_, index=smote_processed_feature_names)

    # Get GB importances from the best tuned GB model with SMOTE
    smote_gb_importances_tuned = pd.Series(best_smote_gb_model.named_steps['classifier'].feature_importances_, index=smote_processed_feature_names)

    # Get XGBoost importances from the best tuned XGBoost model with SMOTE (only if available)
    # This part remains but is now conditional on best_smote_xgb_model existing
    smote_xgb_importances_tuned = None # Initialize to None
    if 'best_smote_xgb_model' in locals() and hasattr(best_smote_xgb_model.named_steps['classifier'], 'feature_importances_'):
         smote_xgb_importances_tuned = pd.Series(best_smote_xgb_model.named_steps['classifier'].feature_importances_, index=smote_processed_feature_names)


    # Calculate mean absolute SHAP value for each feature (as a measure of importance)
    # Using SHAP values calculated from the tuned RF model with SMOTE
    # Need to ensure shap_values_smote is available and correctly structured
    if 'shap_values_smote' in locals() and isinstance(shap_values_smote, list) and len(shap_values_smote) > 1:
        mean_abs_shap_values_smote = np.mean(np.abs(shap_values_smote[1]), axis=0)
        # Ensure the index matches the number of features
        if len(smote_processed_feature_names) == mean_abs_shap_values_smote.shape[0]:
             shap_importance_series_smote = pd.Series(mean_abs_shap_values_smote, index=smote_processed_feature_names)
        else:
             print("Warning: Feature name count mismatch for SHAP importance series. Using generic names.")
             shap_importance_series_smote = pd.Series(mean_abs_shap_values_smote, index=[f'feature_{i}' for i in range(mean_abs_shap_values_smote.shape[0])])

    elif 'shap_values_smote' in locals() and isinstance(shap_values_smote, np.ndarray) and shap_values_smote.ndim == 3:
         # If it's a 3D array, calculate mean abs SHAP for the positive class slice
         mean_abs_shap_values_smote = np.mean(np.abs(shap_values_smote[:, :, 1]), axis=0)
         # Ensure the index matches the number of features
         if smote_processed_feature_names and len(smote_processed_feature_names) == mean_abs_shap_values_smote.shape[0]:
             shap_importance_series_smote = pd.Series(mean_abs_shap_values_smote, index=smote_processed_feature_names)
         else:
             print("Warning: Feature name count mismatch for SHAP importance series (3D array). Using generic names.")
             shap_importance_series_smote = pd.Series(mean_abs_shap_values_smote, index=[f'feature_{i}' for i in range(mean_abs_shap_values_smote.shape[0])])

    elif 'shap_values_smote' in locals() and isinstance(shap_values_smote, np.ndarray) and shap_values_smote.ndim == 2:
         # If it's a 2D array, calculate mean abs SHAP directly
         mean_abs_shap_values_smote = np.mean(np.abs(shap_values_smote), axis=0)
         # Ensure the index matches the number of features
         if smote_processed_feature_names and len(smote_processed_feature_names) == mean_abs_shap_values_smote.shape[0]:
             shap_importance_series_smote = pd.Series(mean_abs_shap_values_smote, index=smote_processed_feature_names)
         else:
             print("Warning: Feature name count mismatch for SHAP importance series (2D array). Using generic names.")
             shap_importance_series_smote = pd.Series(mean_abs_shap_values_smote, index=[f'feature_{i}' for i in range(mean_abs_shap_values_smote.shape[0])])

    else:
         print("SHAP values not available or in unexpected format for summary table calculation.")
         # Set importance to NaN or handle appropriately
         shap_importance_series_smote = pd.Series(np.nan, index=smote_processed_feature_names if smote_processed_feature_names else [f'feature_{i}' for i in range(X_sample_for_shap.shape[1])])


    summary_table_data = {
        'LR_Coefficient': smote_lr_coefficients_tuned,
        'LR_Odds_Ratio': smote_lr_odds_ratios_tuned,
        'RF_Importance': smote_rf_importances_tuned,
        'GB_Importance': smote_gb_importances_tuned,
        'SHAP_Mean_Abs_Importance': shap_importance_series_smote
    }

    # Add XGBoost importance only if it was calculated
    if smote_xgb_importances_tuned is not None:
        summary_table_data['XGBoost_Importance'] = smote_xgb_importances_tuned


    summary_table_df = pd.DataFrame(summary_table_data)

    # Sort by SHAP importance (or another metric if preferred)
    summary_table_df = summary_table_df.sort_values(by='SHAP_Mean_Abs_Importance', ascending=False)

    print("\nFeature Summary Table (with SMOTE - Sorted by SHAP Importance):")
    print(summary_table_df.round(4))

    print("\nNote on Interpretation (SMOTE):")
    print("- Metrics are calculated on models trained with SMOTEd data.")
    print("- 'LR_Coefficient' and 'LR_Odds_Ratio' reflect impact on log-odds/odds on the balanced data distribution.")
    print("- 'RF_Importance', 'GB_Importance' reflect feature utility/impact on the balanced data distribution.")
    print("- 'XGBoost_Importance' reflects feature utility/impact on the balanced data distribution (if XGBoost was used).") # Updated note
    print("  - Compare these results to the non-SMOTE results to see how balancing affected perceived feature importance.")


else:
    print("Could not generate the summary table with SMOTE. Ensure necessary components (LR, RF, GB models, processed feature names, and SHAP values) were successfully obtained.") # Updated error message


print("\n--- ML Model Building and Analysis with SMOTE Complete ---")



# %%
test_case_data = {
    "Age": 23,
    "Weight": 55,
    "Height": 160,
    "Blood Group": 13,
    "Menstrual Cycle Interval": 2,
    "Recent Weight Gain": 1,
    "Skin Darkening": 1,
    "Hair Loss": 1,
    "Acne": 1,
    "Regular Fast Food Consumption": 1,
    "Regular Exercise": 0,
    "Mood Swings": 1,
    "Regular Periods": 0,
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
    predicted_class = best_lr_model.predict(test_case_df)
    print(f"Predicted Class (0: No PCOS, 1: PCOS): {predicted_class[0]}")
except NameError:
    print("Error: 'best_lr_model' is not found. Ensure the model training script has been run.")
except Exception as e:
    print(f"An error occurred during prediction: {e}")


# %%
test_case_data = {
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
    predicted_class = best_lr_model.predict(test_case_df)
    print(f"Predicted Class (0: No PCOS, 1: PCOS): {predicted_class[0]}")
except NameError:
    print("Error: 'best_lr_model' is not found. Ensure the model training script has been run.")
except Exception as e:
    print(f"An error occurred during prediction: {e}")


# %% [markdown]
# 

# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier # Import VotingClassifier
estimators = [
    ('lr', best_lr_model),
    ('rf', best_rf_model),
    ('svm', best_svm_model),
    ('gb', best_gb_model)
]

# Create the Voting Classifier
# 'voting='soft'' uses predicted probabilities, which is generally preferred
voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

print("Training Voting Classifier...")
# Train the Voting Classifier on the original training data (pipeline handles preprocessing)
voting_clf.fit(X_train, y_train)
print("Voting Classifier trained.")

print("\nEvaluating Voting Classifier on Test Set...")
# Predict on the original test data (pipeline handles preprocessing)
y_pred_voting = voting_clf.predict(X_test)
y_proba_voting = voting_clf.predict_proba(X_test)[:, 1]

print("\nClassification Report (Voting Classifier):")
print(classification_report(y_test, y_pred_voting))

print("\nConfusion Matrix (Voting Classifier):")
print(confusion_matrix(y_test, y_pred_voting))

roc_auc_voting = roc_auc_score(y_test, y_proba_voting)
print(f"\nROC AUC Score (Voting Classifier): {roc_auc_voting:.4f}")

# %%
##Model comparison

# %%
df=pd.read_csv('CLEAN- PCOS SURVEY SPREADSHEET.csv')

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

# Separate features (X) and target (Y)
X = df.drop(columns=['PCOS Diagnosis'])
Y = df['PCOS Diagnosis']


# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

print(f"\nTrain features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Train target shape: {Y_train.shape}")
print(f"Test target shape: {Y_test.shape}")

# %%
def calculate_metrics(model, X_test, Y_test):
    """
    Calculates various classification metrics for a given model.

    Args:
        model: Trained machine learning model.
        X_test: Test features.
        Y_test: True test labels.

    Returns:
        A pandas DataFrame containing the calculated metrics.
    """
    try:
        Yhat = model.predict(X_test)
        # Ensure model has predict_proba method for AUC/AUPR
        if hasattr(model, 'predict_proba'):
            Y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(Y_test, Y_proba, pos_label=1)
            roc_auc = roc_auc_score(Y_test, Y_proba)
            aupr = auc(recall, precision)
        else:
            # Handle models without predict_proba (e.g., some SVM kernels without probability=True)
            print(f"Warning: Model {type(model).__name__} does not have predict_proba. AUC and AUPR will be NaN.")
            roc_auc = np.nan
            aupr = np.nan
            # For precision_recall_curve, we need probabilities or decision function scores.
            # If decision_function is available, we could use that, but for simplicity,
            # we'll set AUPR to NaN if predict_proba is missing.
            # If you need AUPR for such models, you'd need a more complex handler.
            precision = np.nan
            recall = np.nan


        metrics = pd.DataFrame(index=['Model'], columns=['Acc', 'Prec', 'Rec', 'F1', 'AUC', 'AUPR'])
        metrics.at['Model', 'Acc'] = accuracy_score(Y_test, Yhat)
        # Handle cases where precision/recall/f1 might be undefined (e.g., no positive predictions)
        metrics.at['Model', 'Prec'] = precision_score(Y_test, Yhat, pos_label=1, zero_division=0)
        metrics.at['Model', 'Rec'] = recall_score(Y_test, Yhat, pos_label=1, zero_division=0)
        metrics.at['Model', 'F1'] = f1_score(Y_test, Yhat, pos_label=1, zero_division=0)
        metrics.at['Model', 'AUC'] = roc_auc
        metrics.at['Model', 'AUPR'] = aupr

        return metrics
    except Exception as e:
        print(f"Error calculating metrics for model {type(model).__name__}: {e}")
        return pd.DataFrame(index=['Model'], columns=['Acc', 'Prec', 'Rec', 'F1', 'AUC', 'AUPR']).fillna(np.nan)


# %%
try:
    # Access the preprocessor and best tuned models from the Robust ML Canvas
    # Define preprocessor structure again if not globally available
    continuous_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    discrete_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', continuous_transformer, cont_col),
            ('cat', discrete_transformer, dis_features_col)
        ],
        remainder='passthrough'
    )

    # Recreate the pipelines if they are not globally available
    # This ensures the preprocessor is included in the pickled pipeline
    pipeline_lr_tuned = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42))])
    pipeline_rf_tuned = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))])
    pipeline_gb_tuned = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier(random_state=42))])
    pipeline_svm_tuned = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(probability=True, class_weight='balanced', random_state=42))])
    # Assuming best_lr_model, best_rf_model, best_gb_model, best_svm_model, voting_clf
    # are available from the Robust ML Canvas run. If not, you'd need to fit them here.

    # Fit the pipelines once on the training data to make them ready for pickling
    # This is necessary if the 'best_..._model' objects from the other script are not pipelines
    # or if you are running this script standalone without running the others first.
    # If you are confident the other scripts leave fitted pipelines as 'best_..._model',
    # you can skip these .fit() calls and pickle the 'best_..._model' directly.
    print("Fitting pipelines for pickling (if necessary)...")
    fitted_pipeline_lr = pipeline_lr_tuned.fit(X_train, Y_train)
    fitted_pipeline_rf = pipeline_rf_tuned.fit(X_train, Y_train)
    fitted_pipeline_gb = pipeline_gb_tuned.fit(X_train, Y_train)
    fitted_pipeline_svm = pipeline_svm_tuned.fit(X_train, Y_train)

    # For the Voting Classifier, it needs the fitted base models (which are pipelines here)
    # Recreate and fit the Voting Classifier pipeline
    estimators_no_smote = [
        ('lr', fitted_pipeline_lr), # Use fitted pipelines as estimators
        ('rf', fitted_pipeline_rf),
        ('svm', fitted_pipeline_svm),
        ('gb', fitted_pipeline_gb)
    ]
    pipeline_voting_tuned = VotingClassifier(estimators=estimators_no_smote, voting='soft', n_jobs=-1)
    fitted_pipeline_voting = pipeline_voting_tuned.fit(X_train, Y_train)


    # Pickle the fitted pipelines
    with open("pcos_tuned_lr_nopipe.pkl", "wb") as file: # Naming convention to distinguish
        pickle.dump(fitted_pipeline_lr, file)
    print("Pickled pcos_tuned_lr_nopipe.pkl")

    with open("pcos_tuned_rf_nopipe.pkl", "wb") as file:
        pickle.dump(fitted_pipeline_rf, file)
    print("Pickled pcos_tuned_rf_nopipe.pkl")

    with open("pcos_tuned_gb_nopipe.pkl", "wb") as file:
        pickle.dump(fitted_pipeline_gb, file)
    print("Pickled pcos_tuned_gb_nopipe.pkl")

    with open("pcos_tuned_svm_nopipe.pkl", "wb") as file:
        pickle.dump(fitted_pipeline_svm, file)
    print("Pickled pcos_tuned_svm_nopipe.pkl")

    with open("pcos_tuned_voting_nopipe.pkl", "wb") as file:
        pickle.dump(fitted_pipeline_voting, file)
    print("Pickled pcos_tuned_voting_nopipe.pkl")

except NameError:
     print("\nWarning: Could not find 'best_..._model' objects from Robust ML Canvas.")
     print("Please run the 'Robust ML Model Building for PCOS Detection' Canvas first.")
except Exception as e:
     print(f"\nError generating pickle files for structured models (no SMOTE): {e}")


# %%
continuous_transformer_smote = Pipeline(steps=[('scaler', StandardScaler())])
discrete_transformer_smote = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
smote_preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer_smote, cont_col),
        ('cat', discrete_transformer_smote, dis_features_col)
    ],
    remainder='passthrough'
)

# Define the imblearn pipelines
pipeline_smote_lr = ImbPipeline(steps=[('preprocessor', smote_preprocessor), ('smote', SMOTE(random_state=42)), ('classifier', LogisticRegression(solver='liblinear', random_state=42))])
pipeline_smote_rf = ImbPipeline(steps=[('preprocessor', smote_preprocessor), ('smote', SMOTE(random_state=42)), ('classifier', RandomForestClassifier(random_state=42))])
pipeline_smote_gb = ImbPipeline(steps=[('preprocessor', smote_preprocessor), ('smote', SMOTE(random_state=42)), ('classifier', GradientBoostingClassifier(random_state=42))])
pipeline_smote_svm = ImbPipeline(steps=[('preprocessor', smote_preprocessor), ('smote', SMOTE(random_state=42)), ('classifier', SVC(probability=True, random_state=42))])
# Corrected XGBoost SMOTE pipeline definition - SMOTE is part of the pipeline, XGBoost is the classifier
# pipeline_smote_xgb = ImbPipeline(steps=[('preprocessor', smote_preprocessor), ('smote', SMOTE(random_state=42)), ('classifier', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42))])


# Fit and Pickle the imblearn pipelines
print("Fitting and pickling imblearn pipelines...")
try:
    fitted_smote_pipeline_lr = pipeline_smote_lr.fit(X_train, Y_train)
    with open("pcos_tuned_smote_lr_pipe.pkl", "wb") as file:
        pickle.dump(fitted_smote_pipeline_lr, file)
    print("Pickled pcos_tuned_smote_lr_pipe.pkl")

    fitted_smote_pipeline_rf = pipeline_smote_rf.fit(X_train, Y_train)
    with open("pcos_tuned_smote_rf_pipe.pkl", "wb") as file:
        pickle.dump(fitted_smote_pipeline_rf, file)
    print("Pickled pcos_tuned_smote_rf_pipe.pkl")

    fitted_smote_pipeline_gb = pipeline_smote_gb.fit(X_train, Y_train)
    with open("pcos_tuned_smote_gb_pipe.pkl", "wb") as file:
        pickle.dump(fitted_smote_pipeline_gb, file)
    print("Pickled pcos_tuned_smote_gb_pipe.pkl")

    fitted_smote_pipeline_svm = pipeline_smote_svm.fit(X_train, Y_train)
    with open("pcos_tuned_smote_svm_pipe.pkl", "wb") as file:
        pickle.dump(fitted_smote_pipeline_svm, file)
    print("Pickled pcos_tuned_smote_svm_pipe.pkl")

    # fitted_smote_pipeline_xgb = pipeline_smote_xgb.fit(X_train, Y_train)
    # with open("pcos_tuned_smote_xgb_pipe.pkl", "wb") as file:
    #     pickle.dump(fitted_smote_pipeline_xgb, file)
    # print("Pickled pcos_tuned_smote_xgb_pipe.pkl")

except Exception as e:
    print(f"\nError fitting or pickling structured models (with SMOTE): {e}")


# %%
loaded_models = {}

# Load your old models
try:
    with open("pcos_randomforest1.pkl", "rb") as file: loaded_models["Old Random Forest v1"] = pickle.load(file)
    print("Loaded old pcos_randomforest1.pkl")
    with open("pcos_decisiontree1.pkl", "rb") as file: loaded_models["Old Decision Tree"] = pickle.load(file)
    print("Loaded old pcos_decisiontree1.pkl")
    with open("pcos_stacking_rf_dt.pkl", "rb") as file: loaded_models["Old Stacking RF + DT"] = pickle.load(file)
    print("Loaded old pcos_stacking_rf_dt.pkl")
    with open("pcos_svm1.pkl", "rb") as file: loaded_models["Old SVM"] = pickle.load(file)
    print("Loaded old pcos_svm1.pkl")
    with open("pcos_stacking_svm_lr.pkl", "rb") as file: loaded_models["Old Stacking SVM + LR"] = pickle.load(file)
    print("Loaded old pcos_stacking_svm_lr.pkl")
    with open("pcos_rf2.pkl", "rb") as file: loaded_models["Old Random Forest v2"] = pickle.load(file)
    print("Loaded old pcos_rf2.pkl")
except FileNotFoundError as e:
    print(f"Warning: Could not load one or more old model pickle files: {e}")
except Exception as e:
    print(f"Error loading old models: {e}")


# Load the newly generated structured models (without SMOTE)
try:
    with open("pcos_tuned_lr_nopipe.pkl", "rb") as file: loaded_models["Tuned LR (No SMOTE, Pipe)"] = pickle.load(file)
    print("Loaded pcos_tuned_lr_nopipe.pkl")
    with open("pcos_tuned_rf_nopipe.pkl", "rb") as file: loaded_models["Tuned RF (No SMOTE, Pipe)"] = pickle.load(file)
    print("Loaded pcos_tuned_rf_nopipe.pkl")
    with open("pcos_tuned_gb_nopipe.pkl", "rb") as file: loaded_models["Tuned GB (No SMOTE, Pipe)"] = pickle.load(file)
    print("Loaded pcos_tuned_gb_nopipe.pkl")
    with open("pcos_tuned_svm_nopipe.pkl", "rb") as file: loaded_models["Tuned SVM (No SMOTE, Pipe)"] = pickle.load(file)
    print("Loaded pcos_tuned_svm_nopipe.pkl")
    with open("pcos_tuned_voting_nopipe.pkl", "rb") as file: loaded_models["Tuned Voting (No SMOTE, Pipe)"] = pickle.load(file)
    print("Loaded pcos_tuned_voting_nopipe.pkl")
except FileNotFoundError as e:
     print(f"Warning: Could not load one or more structured (no SMOTE) model pickle files: {e}")
except Exception as e:
     print(f"Error loading structured models (no SMOTE): {e}")


# Load the newly generated structured models (with SMOTE)
try:
    with open("pcos_tuned_smote_lr_pipe.pkl", "rb") as file: loaded_models["Tuned LR (SMOTE, Pipe)"] = pickle.load(file)
    print("Loaded pcos_tuned_smote_lr_pipe.pkl")
    with open("pcos_tuned_smote_rf_pipe.pkl", "rb") as file: loaded_models["Tuned RF (SMOTE, Pipe)"] = pickle.load(file)
    print("Loaded pcos_tuned_smote_rf_pipe.pkl")
    with open("pcos_tuned_smote_gb_pipe.pkl", "rb") as file: loaded_models["Tuned GB (SMOTE, Pipe)"] = pickle.load(file)
    print("Loaded pcos_tuned_smote_gb_pipe.pkl")
    with open("pcos_tuned_smote_svm_pipe.pkl", "rb") as file: loaded_models["Tuned SVM (SMOTE, Pipe)"] = pickle.load(file)
    print("Loaded pcos_tuned_smote_svm_pipe.pkl")
    # with open("pcos_tuned_smote_xgb_pipe.pkl", "rb") as file: loaded_models["Tuned XGBoost (SMOTE, Pipe)"] = pickle.load(file)
    # print("Loaded pcos_tuned_smote_xgb_pipe.pkl")
except FileNotFoundError as e:
     print(f"Warning: Could not load one or more structured (SMOTE) model pickle files: {e}")
except Exception as e:
     print(f"Error loading structured models (SMOTE): {e}")


# %%
results_list = []

if not loaded_models:
    print("No models were loaded. Cannot perform evaluation.")
else:
    for name, model in loaded_models.items():
        print(f"Evaluating {name}...")
        # Ensure the model is evaluated on the consistent X_test, Y_test
        metrics_df = calculate_metrics(model, X_test, Y_test)
        # Flatten the single-row DataFrame to a dict and add model name
        metrics_dict = metrics_df.iloc[0].to_dict()
        metrics_dict["Model"] = name
        results_list.append(metrics_dict)

    # Create DataFrame from the list of metric dicts
    results_df = pd.DataFrame(results_list)

    # Set 'Model' as the index
    results_df = results_df.set_index("Model")

    # Round for better readability and sort by F1 score (or another preferred metric)
    results_df = results_df.round(4).sort_values(by="F1", ascending=False)

    # Display the results
    print("\n--- Model Performance Comparison on Test Set ---")
    print(results_df)

    # Optional: Display with formatting
    try:
        print("\n--- Model Performance Comparison (Formatted) ---")
        print(results_df.style.background_gradient(cmap='Blues').format("{:.4f}"))
    except Exception as e:
        print(f"Could not apply formatting: {e}")
        print("Displaying raw results DataFrame.")
        print(results_df)

# %%


# %%



