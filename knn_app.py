import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer

import numpy as np

st.title("KNN Classifier Explorer (TF-IDF vs One-Hot Encoding)")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset", df.head())

    text_col = st.selectbox("Select the text column", df.columns)
    label_col = st.selectbox("Select the label column", df.columns)

    encoding_method = st.radio("Select encoding method", ["TF-IDF", "One-Hot"])
    k = st.slider("Select k for KNN", 1, 15, 3, step=2)
    metric = st.selectbox("Select distance metric", ["euclidean", "manhattan", "cosine"])

    X = df[text_col]
    y = df[label_col]

    if encoding_method == "TF-IDF":
        vectorizer = TfidfVectorizer()
        X_encoded = vectorizer.fit_transform(X)
    else:
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    st.subheader("Evaluation Metrics")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1-Score:** {f1:.4f}")
