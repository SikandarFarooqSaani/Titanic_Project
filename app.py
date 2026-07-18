
#python -m pip install --upgrade pip
#pip install numpy pandas matplotlib seaborn scikit-learn streamlit joblib


import pickle
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

@st.cache_resource
def load_artifact(path="titanic_model.pkl"):
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    return artifact


try:
    artifact = load_artifact()
    model = artifact["model"]
    sex_encoder = artifact["sex_encoder"]
    FEATURES = artifact["features"]
    best_source = artifact.get("best_source", "N/A")
except FileNotFoundError:
    st.error(
        "Could not find `titanic_model.pkl`. Please run the training notebook "
        "first and place the exported pickle file in the same folder as this app."
    )
    st.stop()

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Enter a passenger's details below to predict whether they would have "
    "survived the Titanic disaster."
)
st.caption(f"Model in use was selected via: **{best_source}**")

st.divider()


# Sidebar - about

with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a Logistic Regression model trained on the classic "
        "Titanic dataset. Hyperparameters were tuned using GridSearchCV and "
        "RandomizedSearchCV, and the best of the two (by cross-validation "
        "accuracy) was exported and is used here."
    )
    st.write("Features used by the model:")
    st.code("\n".join(FEATURES))

# Input form

with st.form("passenger_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x],
        )
        sex = st.radio("Sex", options=["male", "female"], horizontal=True)
        age = st.slider("Age", min_value=0, max_value=90, value=30)
        fare = st.number_input(
            "Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0
        )

    with col2:
        sibsp = st.number_input(
            "Siblings / Spouses Aboard", min_value=0, max_value=10, value=0, step=1
        )
        parch = st.number_input(
            "Parents / Children Aboard", min_value=0, max_value=10, value=0, step=1
        )
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["S", "C", "Q"],
            format_func=lambda x: {
                "S": "Southampton",
                "C": "Cherbourg",
                "Q": "Queenstown",
            }[x],
        )

    submitted = st.form_submit_button("Predict Survival", use_container_width=True)

# Prediction

if submitted:
    # Sex: label-encoded the same way as training (binary, so this is safe)
    sex_encoded = sex_encoder.transform([sex])[0]

    # Embarked: one-hot encoded the same way as training.
    # Training used pd.get_dummies(..., drop_first=True), which dropped "Embarked_C"
    # and kept "Embarked_Q" and "Embarked_S" as the baseline is C (both 0 means Cherbourg).
    embarked_q = 1 if embarked == "Q" else 0
    embarked_s = 1 if embarked == "S" else 0

    input_df = pd.DataFrame(
        [{
            "Pclass": pclass,
            "Sex": sex_encoded,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked_Q": embarked_q,
            "Embarked_S": embarked_s,
        }]
    )[FEATURES]  # ensure correct column order

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.success(f"### ✅ Survived — probability: {probability:.1%}")
    else:
        st.error(f"### ❌ Did Not Survive — probability of survival: {probability:.1%}")

    st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("See the exact input passed to the model"):
        st.dataframe(input_df)

st.divider()
st.caption(
    "Built with Streamlit • Model trained on the classic Titanic dataset • "
    "For educational purposes only."
)
