# Income Prediction Model Using Adult Census Data

## Project Summary
This project delves into the Adult Census Dataset to predict whether an individual's annual income surpasses $50,000. Using machine learning, data manipulation, and visualization tools, the project builds, trains, and evaluates predictive models with a focus on accuracy and interpretability.

## Key Goal
The main objective is to create a machine learning classifier capable of distinguishing individuals whose incomes exceed $50,000 based on demographic and occupational features like age, education, and hours worked.

## Dataset Overview
The dataset, sourced from the UCI Machine Learning Repository, comprises 48,842 entries and includes both categorical and numerical attributes. Notable features include:

## Tools and Libraries
- **Scikit-Learn**: For implementing machine learning models and model validation.
- **Pandas**: For data wrangling and manipulation.
- **NumPy**: For fast numerical operations.
- **Seaborn**: For advanced statistical visualizations.
- **Matplotlib**: For core plotting and graphical representation.

## Project Workflow

### Data Cleaning and Preparation
1. **Handling Missing Values**: Resolving gaps in data and ensuring consistency.
2. **Encoding Categorical Data**: Converting non-numeric attributes into a format suitable for modeling.
3. **Scaling Features**: Normalizing numeric attributes for model input consistency.

### Exploratory Analysis (EDA)
1. **Feature Exploration**: Visualizing feature distributions and relationships.
2. **Correlation Insights**: Analyzing associations and interactions among variables.
3. **Addressing Class Imbalance**: Evaluating income category distribution and applying adjustments if needed.

### Model Development
Several machine learning models were trained and optimized, including:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **Hyperparameter Tuning**: Conducted to enhance model accuracy and performance.

### Model Evaluation Metrics
Performance of the models was evaluated using:
- **Accuracy**: Overall correctness of the model predictions.
- **Precision, Recall, and F1-Score**: To assess how well the model balances false positives and negatives.
- **Confusion Matrix**: For in-depth classification results.
- **Cross-Validation**: To verify model consistency and robustness.

## Key Findings
The final model was selected for its balanced performance across accuracy, precision, and recall, providing insights into demographic and occupational factors that influence income predictions.

## Potential Next Steps
- Experimenting with alternative algorithms, such as **XGBoost** and **Neural Networks**.
- Introducing additional feature engineering techniques to strengthen predictions.
- Evaluating and addressing any potential bias or fairness concerns.

## Running the Project
1. **Clone the repository**: 
    ```bash
    git clone https://github.com/your-repo/income-census-project.git
    ```

2. **Install required libraries**: 
    ```bash
    pip install -r requirements.txt
    ```

3. **Execute the Notebook or Python script**:
    ```bash
    jupyter notebook
    ```
   **or** 
    ```bash
    python main.py
    ```
