# Project README: PCOS Analysis and Detection

This README provides an overview and execution guide for the two Jupyter notebooks used in the PCOS analysis and detection project: `data_analysis_pcos.ipynb` and `PCOS_detection_model_final.ipynb`.

## 1. `data_analysis_pcos.ipynb`

This notebook focuses on the initial stages of the project, including data loading, cleaning, exploratory data analysis (EDA), and comprehensive statistical analysis to understand the dataset and the relationships between features and Polycystic Ovary Syndrome (PCOS) diagnosis.

**Execution Flow:**

1. **Setup and Configuration:**

   * Imports necessary libraries (pandas, numpy, seaborn, matplotlib, scipy.stats, statsmodels, itertools).

   * Sets up basic configuration, including the path to the dataset file (`pcos_dataset.csv` or `CLEAN- PCOS SURVEY SPREADSHEET.csv`).

   * Defines lists for discrete and continuous column names based on the dataset structure.

2. **Data Loading:**

   * Loads the dataset from the specified CSV file into a pandas DataFrame.

   * Includes error handling in case the file is not found.

   * Prints the shape of the loaded DataFrame.

3. **Data Cleaning and Initial Inspection:**

   * *(Based on typical EDA notebooks, although not explicitly detailed in the provided code snippets for this notebook)* This would typically involve:

     * Standardizing column names for consistency.

     * Checking for and handling missing values (though the provided code doesn't show imputation, this is a standard step).

     * Identifying and removing duplicate rows.

   * Prints descriptive statistics (`.describe()`) for continuous features.

   * Prints value counts (`.value_counts()`) for discrete features, including the target variable 'PCOS Diagnosis', to understand distributions and identify class imbalance.

4. **Exploratory Data Analysis (EDA) and Visualization:**

   * **Continuous vs. Continuous Analysis:**

     * Calculates and prints the Pearson Correlation Matrix for continuous features (`Age`, `Weight`, `Height`).

     * Generates a **Correlation Heatmap** to visualize these correlations.

     * Generates **Pairwise Scatter Plots (Pairplot)** for continuous features, colored by 'PCOS Diagnosis', with KDE plots on the diagonal. This helps visualize relationships between continuous features and how they differ between the two diagnosis groups.

   * **Discrete vs. Discrete Analysis (Chi-Squared Test):**

     * Iterates through all unique pairs of discrete columns.

     * Creates a contingency table (crosstab) for each pair.

     * Performs a **Chi-Squared test of independence** for each pair to determine if there is a statistically significant association between their categories (p < 0.05).

     * Prints the p-value and interpretation for each pair.

     * Collects and summarizes the pairs with statistically significant associations.

   * **Continuous vs. Discrete Analysis (t-tests/ANOVA):**

     * Iterates through selected pairs of continuous and discrete columns (e.g., `Weight` vs `PCOS Diagnosis`, `Age` vs `Blood Group`).

     * Generates **Box Plots** to visualize the distribution of the continuous feature across the categories of the discrete feature.

     * Performs **Independent Samples t-tests** (for binary discrete variables) or **ANOVA** (for discrete variables with >2 categories) to test for statistically significant differences in the mean of the continuous feature across the categories.

     * Prints the p-value and interpretation for each test.

5. **Summary of Analysis:**

   * Prints a summary of the statistically significant discrete feature associations found.

   * Indicates the completion of the feature vs. feature analysis.

**Purpose:**

The primary purpose of this notebook is to gain a deep understanding of the dataset's structure, the distribution of individual features, and the relationships and associations between different features, particularly in relation to the 'PCOS Diagnosis' target variable. The EDA and statistical analysis results inform subsequent steps, such as feature selection, feature engineering, and the choice of appropriate machine learning models and evaluation metrics, especially considering the identified class imbalance.

## 2. `PCOS_detection_model_final.ipynb`

This notebook focuses on the machine learning aspects of the project, including data preprocessing using pipelines, building and evaluating various classification models, handling class imbalance using techniques like `class_weight` and SMOTE, tuning model hyperparameters, and interpreting the final models.

**Execution Flow:**

 1. **Setup and Configuration:**

    * Imports necessary libraries (pandas, numpy, sklearn modules for models, preprocessing, metrics, model selection; imblearn for SMOTE and pipelines; shap for interpretation; matplotlib, seaborn).

    * Sets up basic configuration, including the path to the dataset file.

    * Defines lists for discrete, continuous, and target column names.

    * Suppresses specific warnings.

 2. **Data Loading and Initial Inspection:**

    * Loads the dataset from the specified CSV file into a pandas DataFrame.

    * Includes error handling.

    * Prints the shape of the loaded DataFrame and the distribution of the target variable 'PCOS Diagnosis'.

 3. **Feature/Target Separation and Data Splitting:**

    * Separates the features (X) and the target variable (y).

    * Splits the data into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) using `train_test_split`.

    * Crucially, `stratify=y` is used to ensure that the proportion of PCOS cases is maintained in both the training and testing sets, which is vital for imbalanced datasets.

    * Prints the shapes of the resulting train and test sets.

 4. **Data Preprocessing Pipeline Definition:**

    * Defines preprocessing steps for continuous features (`StandardScaler`) and discrete features (`OneHotEncoder`) using `Pipeline` objects.

    * Combines these transformers using `ColumnTransformer` to apply different transformations to different column subsets.

    * This `preprocessor` object is designed to be integrated into model pipelines.

    * *(Note: The code also includes a step to apply the preprocessor once to the training data to attempt to get processed feature names for interpretability later, but the main workflow uses the preprocessor within model pipelines).*

 5. **Initial Model Training and Cross-Validation (without SMOTE, using `class_weight`):**

    * Defines a dictionary of various classification models (Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN).

    * For models that support it (LR, RF, SVM), `class_weight='balanced'` is used to address class imbalance by giving more weight to the minority class during training.

    * Defines a list of scoring metrics relevant to imbalanced data ('accuracy', 'precision', 'recall', 'f1', 'roc_auc').

    * Performs **5-fold cross-validation** on the *preprocessed training data* (`X_train_processed`) for each initial model.

    * Stores and prints the mean cross-validation scores for each model.

    * Summarizes the cross-validation results, highlighting models that perform well on Recall and F1-Score.

 6. **Model Tuning (using GridSearchCV with Pipelines):**

    * Focuses on tuning the hyperparameters of promising models identified from cross-validation (e.g., Random Forest, Logistic Regression, SVM, Gradient Boosting, XGBoost - if added).

    * For each model, a `Pipeline` is created that includes the `preprocessor` as the first step and the classifier as the second step.

    * A grid of hyperparameters (`param_grid`) is defined for each model.

    * **GridSearchCV** is performed on the *original training data* (`X_train`, `y_train`) using the defined pipeline and parameter grid.

    * The `scoring` parameter in GridSearchCV is set to metrics like 'recall' or 'f1' to optimize for performance on the minority class.

    * GridSearchCV automatically performs cross-validation internally and finds the best combination of hyperparameters.

    * Prints the best parameters found and the best cross-validation score for each tuned model.

    * The `best_estimator_` from GridSearchCV is saved as the best tuned model pipeline.

 7. **Model Training and Cross-Validation (with SMOTE):**

    * *(This section is present in the `pcos_ml_model_smote.ipynb` version)*

    * Defines a dictionary of models.

    * Uses `imblearn.pipeline.Pipeline` to create pipelines that include `preprocessor`, `SMOTE`, and the classifier. SMOTE is applied *only* to the training data within each cross-validation fold.

    * Performs **5-fold cross-validation** on the *original training data* (`smote_X_train`, `smote_y_train`) using these SMOTE-integrated pipelines.

    * Stores and prints the mean cross-validation scores, comparing performance with the `class_weight` approach.

 8. **Model Tuning (using GridSearchCV with SMOTE Pipelines):**

    * *(This section is present in the `pcos_ml_model_smote.ipynb` version)*

    * Similar to step 6, but uses the `ImbPipeline` with SMOTE integrated.

    * Performs GridSearchCV on the original training data using the SMOTE-integrated pipelines, optimizing for Recall or F1.

 9. **Final Model Evaluation on Test Set:**

    * Evaluates the **best tuned models** (from both `class_weight` and SMOTE approaches) on the completely **untouched test set** (`X_test`, `y_test` or `smote_X_test`, `smote_y_test`).

    * The trained pipelines (which include preprocessing) are used for prediction.

    * Calculates and prints the **Classification Report** (showing precision, recall, f1-score, support for each class) and the **Confusion Matrix**.

    * Calculates and prints the **ROC AUC Score** and **AUPR Score**.

    * These metrics on the test set provide the final, unbiased assessment of model performance.

10. **Feature Importance and Impact Analysis:**

    * Analyzes the trained models to understand which features were most influential in the predictions.

    * **Logistic Regression Coefficients:** Extracts and sorts coefficients from the tuned LR model (trained on scaled/one-hot encoded data). Calculates and interprets Odds Ratios.

    * **Tree-based Importances:** Extracts and sorts `feature_importances_` from tuned Random Forest and Gradient Boosting models.

    * **SHAP Values:** Uses the `shap` library (e.g., `shap.TreeExplainer` for tree models) to calculate SHAP values for a sample of the training data.

    * Generates **SHAP Summary Plots** (bar plot for overall importance, dot plot for impact and direction) to visualize feature influence.

    * Interprets the SHAP plots, explaining how feature values push predictions towards or away from the positive class.

    * *(The code also attempts to simulate a combined feature summary table showing different importance metrics).*

11. **Saving the Final Model:**

    * *(Based on the `PCOS_detection_model_final.ipynb` notebook)*

    * The final chosen model pipeline (e.g., the Stacking SVM + LR model) is saved to a file (`pcos_stacking_svm_lr.pkl`) using the `pickle` library. This allows the trained model to be loaded and used later for making new predictions without retraining.

**Purpose:**

This notebook's purpose is to implement and evaluate various machine learning algorithms for PCOS detection based on the preprocessed lifestyle and related features. It demonstrates how to handle class imbalance effectively, tune models for optimal performance on relevant metrics (Recall, F1), and interpret the resulting models to understand the contribution of different features to the predictions. The outcome is a set of evaluated models and potentially a final selected model saved for deployment.

This README provides a high-level overview of the steps and purpose of each notebook. For detailed code implementation and exact outputs, please refer to the respective `.ipynb` files.