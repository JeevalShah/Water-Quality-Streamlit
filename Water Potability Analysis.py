import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("water_rfc_model.pkl")

st.title("💧 Water Potability Prediction")

st.write("""
Enter water quality parameters to predict whether the water is potable (safe to drink) or not.
""")

# User inputs
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness (mg/L)", value=100.0)
solids = st.number_input("Solids (ppm)", value=10000.0)
chloramines = st.number_input("Chloramines (mg/L)", value=5.0)
sulfate = st.number_input("Sulfate (mg/L)", value=250.0)
conductivity = st.number_input("Conductivity (μS/cm)", value=300.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", value=10.0)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=60.0)
turbidity = st.number_input("Turbidity (NTU)", value=3.0)

# Predict button
if st.button("Predict"):
    features = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity,
                          organic_carbon, trihalomethanes, turbidity]])
    
    # Predict class and probability
    pred_class = model.predict(features)[0]
    pred_prob = model.predict_proba(features)[0]
    
    # Determine color and majority confidence
    if pred_class == 1:
        box_color = "#d4edda"  # light green
        border_color = "#28a745"  # dark green
        confidence = pred_prob[1]
        class_text = "Potable 💧"
    else:
        box_color = "#f8d7da"  # light red
        border_color = "#dc3545"  # dark red
        confidence = pred_prob[0]
        class_text = "Not Potable 🚫"
    
    # Display prediction in colored box
    st.markdown(
        f"""
        <div style="
            background-color: {box_color};
            border: 2px solid {border_color};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        ">
            <h2 style='color:black'>{class_text}</h2>
            <p style='color:black'>Confidence: {confidence:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
