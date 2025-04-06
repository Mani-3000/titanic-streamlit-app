import streamlit as st
import numpy as np
import joblib

model = joblib.load("logistic_model.pkl")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, step=1)
parch = st.number_input("Parents/Children Aboard", min_value=0, step=1)
fare = st.number_input("Fare", min_value=0.0, step=1.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

sex_encoded = 1 if sex == "male" else 0
embarked_dict = {"S": 2, "C": 0, "Q": 1}
embarked_encoded = embarked_dict[embarked]

input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.success(f"🎉 Likely to survive! (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Unlikely to survive. (Probability: {prob:.2f})")
