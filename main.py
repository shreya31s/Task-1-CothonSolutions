import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


# Load dataset
df = pd.read_csv("diabetes.csv")

# Display basic details
print(df.head())  
print(df.info())  
print(df.describe())  

#Check for Missing values
print(df.isnull().sum)
df.fillna(df.mean(), inplace=True)
#Remove duplicates if found
print(df.duplicated().sum())  
df.drop_duplicates(inplace=True)  

#Check for outliers(Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

#Convert categorical features 
df = pd.get_dummies(df, drop_first=True)  

#Feature selection
#Check correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

#Recursive feature elimination(RFE)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X, y)

selected_features = X.columns[fit.support_]
print("Selected Features:", selected_features)

df = df[selected_features]
df["Outcome"] = y

#Data visualization(EDA)
#pair plot
sns.pairplot(df, hue="Outcome")
plt.show()

#distribution of outcome variable
sns.countplot(x=df["Outcome"])
plt.show()

#feature distribution
df.hist(figsize=(10, 8))
plt.show()

#model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} Model Trained Successfully")
    
#model evaluation
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} Model Evaluation:")
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

#save best model
best_model = RandomForestClassifier()  
best_model.fit(X_train, y_train)  
joblib.dump(best_model, "diabetes_prediction_model.pkl") 