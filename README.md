# Heart-Disease-Prediction
Predicting heart disease using machine learning with comprehensive data analysis and visualization.

## Overview

This project aims to predict the presence of heart disease in patients using a variety of features provided in a dataset. The analysis includes data preprocessing, exploratory data analysis (EDA), and application of machine learning models to achieve accurate predictions.

## Dataset

The dataset used in this project contains several features relevant to heart disease prediction, such as age, sex, chest pain type, resting blood pressure, cholesterol levels, and more. It is loaded from a CSV file named `Heart-Disease.csv`.

## Setup Instructions

To run this project, you need to have Python and several libraries installed. Follow the steps below to set up the environment:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```
2. **Install required packages:**

You can install the required packages using `pip`:

```sh
pip install -r requirements.txt
```
Alternatively, you can manually install the packages used in this project:

```python
pip install numpy pandas matplotlib seaborn scikit-learn
```
3. **Run the Jupyter Notebook:**

Launch Jupyter Notebook and open the Heart Disease Prediction.ipynb file:

```sh
jupyter notebook Heart\ Disease\ Prediction.ipynb
```
## Exploratory Data Analysis (EDA)
The EDA section includes the following steps:

- Data Loading and Inspection:

The dataset is loaded using pandas, and basic inspections such as checking the first few rows, data types, and missing values are performed.

```python
df = pd.read_csv("Heart-Disease.csv")
df.head()
df.info()
```
- Statistical Summary:

A statistical summary of the dataset is generated to understand the distribution of features.

```python
df.describe()
```

- Visualization:

Histograms and a correlation heatmap are created to visualize the distribution of features and their relationships.

```python
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    plt.hist(df[feature], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")
plt.tight_layout()
plt.show()
```
```python
plt.figure(figsize=(16, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Correlation Heatmap')
plt.show()
```

## Data Preprocessing
- Normalization:

Some features are log-transformed to reduce skewness and improve model performance.

```python
df['trestbps'] = np.log(df['trestbps'])
df['chol'] = np.log(df['chol'])
df['thalach'] = np.log(df['thalach'])
```

## Machine Learning Models
The project applies various machine learning models to predict heart disease. The dataset is split into training and testing sets, and models such as Logistic Regression, Decision Trees, and Random Forests are used.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example with Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
```

## Conclusion
This project demonstrates the process of predicting heart disease using data science techniques, from data preprocessing and EDA to model training and evaluation. The results highlight the importance of feature selection and model choice in achieving accurate predictions.

## Acknowledgements
This project uses a publicly available dataset and builds upon standard data science and machine learning practices. Special thanks to the data providers and the open-source community.

