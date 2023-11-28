import streamlit as st
import numpy as np
import joblib

# Set page title and configure layout
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon=":microscope:",
    layout="wide",
)

# Specify the absolute paths to the model and scaler files using raw strings
model_file_path = r"C:\Users\mksuv\logistic_regression_model.pkl"
scaler_file_path = r"C:\Users\mksuv\scaler.pkl"

# Load the pre-trained logistic regression model and scaler
#st.write("Loading model...")
logistic_regression_model = joblib.load(model_file_path)
#st.write("Model loaded successfully.")

#st.write("Loading scaler...")
scaler = joblib.load(scaler_file_path)
#st.write("Scaler loaded successfully.")



# Function to make predictions
def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, total_proteins, albumin, ag_ratio, sgpt, sgot, alkphos):
    input_data = np.array([[age, gender, total_bilirubin, direct_bilirubin, total_proteins, albumin, ag_ratio, sgpt, sgot, alkphos]])
    input_data = scaler.transform(input_data)
    #st.write("Transformed Input Data:", input_data)
    prediction = logistic_regression_model.predict(input_data)
    #st.write("Raw Prediction:", prediction)
    return prediction[0]

# UI/UX Design
st.title("Liver Disease Prediction App")
st.image("https://img.onmanorama.com/content/dam/mm/en/wellness/health/images/2017/4/26/liver.jpg", use_column_width=True)

# Introduction Section
st.markdown(
    """
    Welcome to the Liver Disease Prediction App! 
    This app predicts the likelihood of liver disease based on patient details.
    Adjust the sliders in the sidebar and click 'Predict' to get the prediction.
    """
)

# Sidebar for User Input
st.sidebar.header("Patient Details")
age = st.sidebar.slider("Age", 0, 100, 25)
gender = st.sidebar.radio("Gender", ["Female", "Male"])
total_bilirubin = st.sidebar.slider("Total Bilirubin", 0.1, 45.0, 1.0)
direct_bilirubin = st.sidebar.slider("Direct Bilirubin", 0.0, 20.0, 0.5)
total_proteins = st.sidebar.slider("Total Proteins", 100, 2500, 500)
albumin = st.sidebar.slider("Albumin", 10, 2000, 100)
ag_ratio = st.sidebar.slider("AG Ratio", 10, 5000, 2000)
sgpt = st.sidebar.slider("SGPT", 3.0, 8.0, 5.0)
sgot = st.sidebar.slider("SGOT", 1.0, 5.0, 2.5)
alkphos = st.sidebar.slider("Alkaline Phosphatase", 0.0, 2.0, 1.0)

# Prediction button
if st.sidebar.button("Predict", key="predict_button", help="Click to predict liver disease"):
    # Convert gender to numerical value
    gender = 0 if gender == "Female" else 1
    
    # Make prediction
    result = predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, total_proteins, albumin, ag_ratio, sgpt, sgot, alkphos)
    st.success(f"The likelihood of liver disease is: {result}")

# Additional Information
st.sidebar.markdown("## Additional Information")
st.sidebar.info(
    "The prediction is based on a logistic regression model trained on liver patient data."
    " Always consult with healthcare professionals for accurate diagnoses."
)

# About Section
st.markdown("## About")
st.write(
    """
    This app uses a logistic regression model to predict the likelihood of liver disease.
    The model was trained on a dataset of liver patient records.
    """
)

# Contact Us Section
st.markdown("## Contact Us")
st.write(
    """
    If you have any questions or concerns, please reach out to us:
    
    :email: Email: [liverapp@example.com](mailto:liverapp@example.com)
    :phone: Phone: +123 456 7890
    """
)
