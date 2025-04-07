import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load("logistic_model.pkl")

# Streamlit app UI
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# User input for the model
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
fare = st.number_input("Fare", min_value=0.0, step=1.0)

# Encoding categorical variables
sex_encoded = 1 if sex == "male" else 0

# Preparing input data for prediction (only using 4 features)
input_data = np.array([[pclass, sex_encoded, age, fare]])

# Prediction logic
if st.button("Predict"):
    # Ensure input_data has the correct shape (1, 4) for the model
    if input_data.shape[1] == 4:  # Model expects 4 features
        try:
            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]  # Probability of survival

            # Display the prediction results
            if prediction == 1:
                st.success(f"🎉 Likely to survive! (Probability: {prob:.2f})")
            else:
                st.error(f"❌ Unlikely to survive. (Probability: {prob:.2f})")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error(f"Error: Expected input shape (1, 4), but got {input_data.shape}")
