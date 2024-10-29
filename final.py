import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("new data with target class.csv")
    return df

# Load dataset
df = load_data()

# Display dataset info
st.title("Arrhythmia Classification Analysis")
st.write("### Dataset Overview")
st.write(df.head())
st.write("### Summary Statistics")
st.write(df.describe())
st.write("### Total Number of Null Values")
st.write(pd.isnull(df).sum().sum())

# Visualizations
st.write("## Data Visualizations")

# Countplot for class distribution
st.subheader("Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x='class', data=df, ax=ax)
st.pyplot(fig)

# Pie chart for class distribution
st.subheader("Class Distribution (Log-Normalized)")
values = df["class"].value_counts(sort=False).tolist()
labels = df["class"].unique()
log_norm = [np.log10(i + 1) for i in values]

fig, ax = plt.subplots()
patches = ax.pie(log_norm, autopct='%1.1f%%', startangle=90)
ax.legend(loc='best', labels=[f'{l}, {s:.1f}%' for l, s in zip(labels, log_norm)])
ax.axis('equal')
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), ax=ax)
st.pyplot(fig)

# Feature distributions
st.subheader("Boxplot of Selected Features")
features = ["QRS_Dur", "P-R_Int", "Q-T_Int", "T_Int", "P_Int"]
fig, ax = plt.subplots()
sns.boxplot(data=df[features], ax=ax)
st.pyplot(fig)

# Model Training and Evaluation
st.write("## Model Training and Evaluation")

# Split data
X = df.drop(columns="class")
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "KNN Classifier": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(solver='saga'),
    "Decision Tree": DecisionTreeClassifier(),
    "Linear SVC": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=300)
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_recall = recall_score(y_train, y_pred_train, average="weighted")
    test_recall = recall_score(y_test, y_pred_test, average="weighted")

    results.append({
        "Model": model_name,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Train Recall": train_recall,
        "Test Recall": test_recall
    })

# Display results
results_df = pd.DataFrame(results)
st.write("### Model Evaluation Results")
st.write(results_df)

# Confusion matrix for the last model
st.subheader("Confusion Matrix for Random Forest")
y_pred = models["Random Forest"].predict(X_test)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Footer
st.write("### Thank you for using the Arrhythmia Classification Analysis App!")
classifiers = {
    'KNN Classifier': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Linear SVC': LinearSVC()
}

# Dummy model training step (replace with your trained models)
for clf in classifiers.values():
    clf.fit(X_train, y_train)  # Assuming X_train, y_train are defined

# Create a function to make predictions
def predict(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = scaler.transform(input_df)  # Scale input
    prediction = clf.predict(input_df)  # Use the classifier to predict
    return prediction


# Load the trained model and scaler
model = joblib.load('KSVC_clf.pkl')

# List of all 121 feature names
feature_names = [
    'Age', 'Sex', 'Height', 'Weight', 'AVF00', 'AVF01', 'AVF02', 'AVF03', 'AVF04', 'AVF05',
    'AVF06', 'AVF07', 'AVF08', 'AVF09', 'AVF10', 'AVF11', 'AVF12', 'AVF13', 'AVF14', 'AVF15',
    'AVF16', 'AVF17', 'AVF18', 'AVF19', 'AVF20', 'AVF21', 'AVF22', 'AVF23', 'AVF24', 'AVF25',
    'AVF26', 'AVF27', 'AVF28', 'AVF29', 'AVF30', 'AVF31', 'AVF32', 'AVF33', 'AVF34', 'AVF35',
    'AVF36', 'AVF37', 'AVF38', 'AVF39', 'AVF40', 'AVF41', 'AVF42', 'AVF43', 'AVF44', 'AVF45',
    'AVF46', 'AVF47', 'AVF48', 'AVF49', 'AVF50', 'AVF51', 'AVF52', 'AVF53', 'AVF54', 'AVF55',
    'AVF56', 'AVF57', 'AVF58', 'AVF59', 'AVF60', 'AVF61', 'AVF62', 'AVF63', 'AVF64', 'AVF65',
    'AVF66', 'AVF67', 'AVF68', 'AVF69', 'AVF70', 'AVF71', 'AVF72', 'AVF73', 'AVF74', 'AVF75',
    'AVF76', 'AVF77', 'AVF78', 'AVF79', 'AVF80', 'AVF81', 'AVF82', 'AVF83', 'AVF84', 'AVF85',
    'AVF86', 'AVF87', 'AVF88', 'AVF89', 'AVF90', 'AVF91', 'AVF92', 'AVF93', 'AVF94', 'AVF95',
    'AVF96', 'AVF97', 'AVF98', 'AVF99', 'AVF100', 'AVF101', 'AVF102', 'AVF103', 'AVF104', 'AVF105',
    'AVF106', 'AVF107', 'AVF108', 'AVF109', 'AVF110', 'AVF111', 'AVF112', 'AVF113', 'AVF114', 'AVF115',
    'AVF116'
]


# Create input form for user input
def user_input_features():
    inputs = {}
    for feature in feature_names:
        if feature in ['Sex']:  # Example of a categorical variable
            inputs[feature] = st.selectbox(feature, options=['male', 'female'])
        else:
            inputs[feature] = st.number_input(feature)
    return pd.DataFrame(inputs, index=[0])
# Mapping from integer predictions to class labels
class_names = [
    "Normal", 
    "Ischemic changes (CAD)", 
    "Old Anterior Myocardial Infarction",
    "Old Inferior Myocardial Infarction",
    "Sinus tachycardia", 
    "Sinus bradycardia", 
    "Ventricular Premature Contraction (PVC)",
    "Supraventricular Premature Contraction",
    "Left Bundle Branch Block",
    "Right Bundle Branch Block",
    "1st Degree Atrioventricular Block",
    "2nd Degree AV Block",
    "3rd Degree AV Block",
    "Left Ventricular Hypertrophy",
    "Atrial Fibrillation or Flutter",
    "Others"
]

# Create a mapping from integer predictions to class labels
class_mapping = {i + 1: class_names[i] for i in range(len(class_names))}

# Function to predict and return the class label
def predict(input_data):
    input_data['Sex'] = input_data['Sex'].map({'male': 0, 'female': 1})  # Convert gender to numeric
    prediction = model.predict(input_data)  # Make prediction
    predicted_class = class_mapping[prediction[0]]  # Map prediction to class label
    return predicted_class

# Streamlit app layout
st.title('Arrhythmia Classification')
input_data = user_input_features()

if st.button('Predict'):
    predicted_class = predict(input_data)
    st.write('Prediction:', predicted_class)


