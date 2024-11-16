import streamlit as st
import webbrowser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
   
    st.sidebar.image("Images/2.png", use_column_width=False, width =200)  
    selected_page = st.sidebar.radio("Navigation Menu", ["Home",  "Loan Prediction"])

    if selected_page ==  "Loan Prediction":
        show_loan_predictor()

    elif selected_page == "Home":
        show_home_page()

def show_home_page():        
    image_path = 'Images\LOAN PREDICTOR.jpg'
    st.image(image_path, use_column_width=True)
    st.subheader("Empowering Your Lending Decisions with Precision")
    html_content = """
    <div style="font-size: 18px; line-height: 1.5;">
        <p>Welcome to Loan Status Predictor! Our advanced algorithms analyze loan applications in seconds, providing accurate predictions to help lenders make confident decisions.</p>
        <ol>
            <li><b>Instant Predictions</b>: Get real-time insights into the likelihood of loan approval or rejection.</li>
            <li><b>Customizable Models</b>: Tailor the prediction algorithms to fit your specific lending criteria.</li>
            <li><b>Intuitive Interface</b>: Easy-to-use dashboard and tools for seamless navigation and analysis.</li>
            <li><b>Interactive Reports</b>: Visualize data and trends with interactive charts and graphs.</li>
        <h3><center><b>Try Our Predictor Now!</b></center></h3>
    </div>
    """
    st.markdown(html_content,unsafe_allow_html=True)

def show_loan_predictor():
    # Load and preprocess data
    @st.cache_data
    def load_and_preprocess_data():
        loan_dataset = pd.read_csv('sample.csv.csv').dropna()
        loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}, 'Married': {'No': 0, 'Yes': 1}, 'Gender': {'Male': 1, 'Female': 0},
                          'Self_Employed': {'No': 0, 'Yes': 1}, 'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                          'Education': {'Graduate': 1, 'Not Graduate': 0}, 'Dependents': {'3+': 4}}, inplace=True)
        X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
        Y = loan_dataset['Loan_Status']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        X_pca = pca.fit_transform(X_scaled)
        X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.1, stratify=Y, random_state=2)
        return X_train, X_test, Y_train, Y_test, scaler, pca

    X_train, X_test, Y_train, Y_test, scaler, pca = load_and_preprocess_data()

# Train classifier
    @st.cache_data
    def train_classifier():
        classifier = svm.SVC(kernel='linear')
        classifier.fit(X_train, Y_train)
        return classifier

    classifier = train_classifier()

# Streamlit UI
    st.title("Loan Status Prediction")
    gender = st.selectbox("Please select your gender", ["Male", "Female"])
    married = st.selectbox("Please select your Marital status", ["Married", "Unmarried"])
    dependents = st.selectbox("Please select number of dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Please select your Education", ["Not Graduate", "Graduate"])
    self_employed = st.selectbox("Please select your Employment status", ["Unemployed", "Employed"])
    applicant_income = st.number_input("Enter your Annual Income (0 to 1 lakh rupees)", value=0, step=1, format="%d")
    coapplicant_income = st.number_input("Enter Co-applicant Income (0 to 50 thousand rupees)", value=0, step=1, format="%d")
    loan_amount = st.number_input("Enter Loan Amount (5 thousand to 10 lakh rupees)", value=0, step=1, format="%d")
    loan_amount_term = st.number_input("Enter Loan Amount Term (12 months to 480 months)", value=0, step=1, format="%d")
    credit_history = st.selectbox("Please select your Credit History", ["No", "Yes"])
    property_area = st.selectbox("Please select your Property Area", ["Rural", "Semiurban", "Urban"])

    if st.button("Predict"):
        gender_val = 1 if gender == "Male" else 0
        married_val = 1 if married == "Married" else 0
        if dependents == '3+':
            dependents_val = 4
        else:
            dependents_val = int(dependents)
        education_val = 1 if education == "Graduate" else 0
        self_employed_val = 1 if self_employed == "Employed" else 0
        credit_history_val = 1 if credit_history == "Yes" else 0
        property_area_val = ["Rural", "Semiurban", "Urban"].index(property_area)

        input_data = np.array([[gender_val, married_val, dependents_val, education_val, self_employed_val, applicant_income,
                            coapplicant_income, loan_amount, loan_amount_term, credit_history_val, property_area_val]])

        input_data_scaled = scaler.transform(input_data)
        input_data_pca = pca.transform(input_data_scaled)

        prediction = classifier.predict(input_data_pca)
        st.header("Prediction")
        st.write("Result:", "You are eligible for a loan" if prediction[0] == 1 else "You are not eligible for a loan")

if __name__ == "__main__":
    main()





