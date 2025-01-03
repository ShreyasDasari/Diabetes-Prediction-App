import streamlit as st
import h2o
from h2o.frame import H2OFrame
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.graph_objects as go
import plotly.express as px

# Initialize H2O cluster
@st.cache_resource
def init_h2o():
    h2o.init()
    return True

# Load Model Functionality
@st.cache_resource
def load_model():
    try:
        # Initialize H2O
        init_h2o()
        # Load the saved H2O model
        return h2o.load_model('./automlmodel')  # Ensure the correct path
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
model = load_model()

if model is None:
    st.stop()  # Stop execution if the model couldn't be loaded

# Get feature names dynamically from the model
try:
    model_features = [feature for feature in model._model_json['output']['names'] if feature != 'is_diab']  # Exclude target column
except KeyError:
    st.error("Unable to retrieve feature names from the model. Please check the model file.")
    st.stop()

if not model_features:
    st.error("No features found in the model. Ensure the model was trained correctly.")
    st.stop()

# Sidebar for User Inputs
st.sidebar.header('Input Features')
def user_input_features():
    input_data = {}
    for feature in model_features:  # Adapted to H2O model
        # Create a slider or text input for each feature dynamically
        if feature.lower() == "glucose":
            input_data[feature] = st.sidebar.slider(f"{feature}", 0, 400, 100)  # Adjusted slider range
        elif feature.lower() == "age":
            input_data[feature] = st.sidebar.slider(f"{feature}", 0, 100, 30)  # Adjusted slider range
        elif feature.lower() not in ["gender_female", "gender_male"]:
            input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0, step=0.1)
    gender = st.sidebar.radio("Gender", ("Male", "Female"), key="gender_radio")
    input_data["gender_female"] = 1 if gender == "Female" else 0
    input_data["gender_male"] = 1 if gender == "Male" else 0
    return pd.DataFrame([input_data])

# Section 1: User Input and Single Prediction
st.header('Diabetes Prediction App')
st.markdown('''
### Empowering Early Detection of Diabetes for a Healthier Future
Diabetes is a global health challenge affecting millions of individuals worldwide. Early detection is crucial for managing the disease and preventing severe complications such as cardiovascular diseases, kidney failure, and vision loss. This interactive tool leverages cutting-edge machine learning technology to help users assess their risk of diabetes in just a few clicks.

With this app, you can:
- **Understand Your Health Profile**: Enter your diagnostic measures and get a personalized prediction based on your unique health parameters.
- **Take Action**: Use the insights provided to consult healthcare professionals for early interventions.
- **Batch Processing for Healthcare Providers**: Upload datasets for multiple individuals to streamline patient risk assessments.

The app also provides **visual insights**, including feature importance and personalized risk visualizations, to help you better understand the factors influencing your prediction.

**Why Use This Tool?**
- **Fast & Accurate**: Powered by advanced machine learning models with high accuracy.
- **Easy-to-Use Interface**: A seamless experience for individuals and healthcare providers alike.
- **Comprehensive Insights**: Detailed analytics to help guide your health decisions.

Explore the codebase and contribute to this open-source initiative:  
[GitHub Repository](https://github.com/ShreyasDasari/Diabetes-Prediction-App)  

Start your journey toward better health today!
''')

# Collect user input dynamically
user_data = user_input_features()

# Display User Inputs
st.subheader('User Input Features')
st.markdown('Provide diagnostic measures to predict whether a patient is diabetic.')
st.write(user_data)

# Prediction
if st.button('Predict'):
    try:
        if user_data.isnull().values.any():
            st.error("All fields must be filled out before prediction.")
        else:
            user_data_h2o = H2OFrame(user_data)  # Convert to H2OFrame
            # Explicitly set columns as categorical based on training data
            for col, domain in zip(model_features, model._model_json['output']['domains']):
                if domain is not None:  # Column is categorical
                    user_data_h2o[col] = user_data_h2o[col].asfactor()
            prediction = model.predict(user_data_h2o).as_data_frame()
            st.subheader('Prediction Result')
            st.write(f"**{'Diabetes' if prediction['predict'][0] == 1 else 'No Diabetes'}** (0: No Diabetes, 1: Diabetes)")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Section 2: Batch Predictions and File Upload
st.header('Batch Prediction and Analysis')
st.markdown('Upload a CSV file to predict outcomes for multiple patients.')

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    try:
        # Read and display uploaded data
        data = pd.read_csv(uploaded_file)
        st.subheader('Uploaded Data Preview')
        st.write(data)

        # Check for missing columns and align with model features
        missing_features = [feature for feature in model_features if feature not in data.columns]
        if missing_features:
            st.error(f"The uploaded data is missing the following features: {missing_features}")
        else:
            # Exclude target column if accidentally included
            if 'is_diab' in data.columns:
                data = data.drop(columns=['is_diab'])

            # Predict outcomes for batch data
            data_h2o = H2OFrame(data)  # Convert to H2OFrame
            # Explicitly set columns as categorical based on training data
            for col, domain in zip(model_features, model._model_json['output']['domains']):
                if domain is not None:  # Column is categorical
                    data_h2o[col] = data_h2o[col].asfactor()
            predictions = model.predict(data_h2o).as_data_frame()

            # Combine predictions with input data
            data['Prediction'] = predictions['predict']
            st.subheader('Predictions')
            st.write(data)

            # Download button for predictions
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

# Section 3: Feature Importance Visualization
st.header('Feature Importance Visualization')
if hasattr(model, 'varimp'):
    varimp = model.varimp()
    importance_df = pd.DataFrame(varimp, columns=['Feature', 'Relative Importance', 'Scaled Importance', 'Percentage']).sort_values(by='Percentage', ascending=False)

    # Plot feature importance
    st.subheader('Feature Importance')
    fig, ax = plt.subplots()
    sns.barplot(x='Percentage', y='Feature', data=importance_df, ax=ax, palette='coolwarm')
    ax.set_title('Feature Importance')
    st.pyplot(fig)
else:
    st.write("Feature importance is not available for the current model.")

# Section 4: Batch Data Analysis
if uploaded_file:
    st.header('Batch Data Visualization')
    st.markdown('Analyze the uploaded dataset with interactive visualizations.')

    # Display histogram for numerical columns
    st.subheader('Distribution of Numerical Features')
    try:
        # Ensure that the uploaded data has numeric columns
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.intersection(model_features)
        if len(numeric_columns) == 0:
            st.write("No numerical features available for visualization.")
        else:
            for column in numeric_columns:
                fig, ax = plt.subplots()
                sns.histplot(data[column], kde=True, ax=ax, color="skyblue")
                ax.set_title(f'Distribution of {column}')
                st.pyplot(fig)

        # Pairplot visualization for feature relationships
        st.subheader("Feature Relationships (Pairplot)")
        pairplot_features = st.multiselect(
            "Select features for pairplot visualization (max 5):",
            options=numeric_columns,
            default=numeric_columns[:5] if len(numeric_columns) > 0 else []
        )
        if len(pairplot_features) > 1:
            sns_pairplot = sns.pairplot(data[pairplot_features], palette='coolwarm')
            st.pyplot(sns_pairplot)
        else:
            st.write("Please select at least two features for pairplot visualization.")

    except Exception as e:
        st.error(f"Error during visualization: {e}")

# Additional Visualizations for User Inputs
# Radar Chart for User Profile
if st.button('Radar Chart (Feature Profile)'):
    try:
        diabetic_avg = [200, 35, 50, 80, 120]
        non_diabetic_avg = [100, 25, 40, 70, 100]
        user_values = user_data.iloc[0].tolist()

        features = model_features[:5]
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(r=diabetic_avg, theta=features, fill='toself', name='Diabetic Profile'))
        fig.add_trace(go.Scatterpolar(r=non_diabetic_avg, theta=features, fill='toself', name='Non-Diabetic Profile'))
        fig.add_trace(go.Scatterpolar(r=user_values, theta=features, fill='none', name='User Input'))

        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error generating radar chart: {e}")

# Prediction Confidence Gauge
if st.button('Prediction Confidence Gauge'):
    try:
        # Convert user data to H2OFrame and align columns
        user_data_h2o = H2OFrame(user_data)
        user_data_h2o.columns = model_features  # Align columns
        for col, domain in zip(model_features, model._model_json['output']['domains']):
            if domain is not None:  # Column is categorical
                user_data_h2o[col] = user_data_h2o[col].asfactor()

        # Get probabilities from prediction
        prediction = model.predict(user_data_h2o).as_data_frame()
        probabilities = prediction.iloc[0, 1:]  # Extract probability values (columns other than 'predict')

        # Use the probability of "Diabetes" as the confidence value
        diabetes_probability = probabilities[1]  # Assuming the second column corresponds to 'Diabetes'
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=diabetes_probability,
            title={'text': "Prediction Confidence"},
            gauge={
                'axis': {'range': [0, 1]},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 1], 'color': "tomato"}
                ],
                'bar': {'color': "darkblue"}
            }
        ))
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error generating gauge chart: {e}")
