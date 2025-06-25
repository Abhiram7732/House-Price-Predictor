import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set page config
st.set_page_config(
    page_title="Boston Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and model
@st.cache_data
def load_data():
    data = pd.read_csv('boston_house_prices.csv', header=1)
    return data

@st.cache_resource
def load_model():
    data = load_data()
    X = data.drop(columns='MEDV')
    y = data['MEDV']
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y

data = load_data()
model, X, y = load_model()

# Calculate model metrics
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

# Feature descriptions
feature_descriptions = {
    'CRIM': 'Per capita crime rate by town',
    'ZN': 'Proportion of residential land zoned for lots over 25,000 sq.ft.',
    'INDUS': 'Proportion of non-retail business acres per town',
    'CHAS': 'Charles River dummy variable (1 if tract bounds river; 0 otherwise)',
    'NOX': 'Nitric oxides concentration (parts per 10 million)',
    'RM': 'Average number of rooms per dwelling',
    'AGE': 'Proportion of owner-occupied units built prior to 1940',
    'DIS': 'Weighted distances to five Boston employment centres',
    'RAD': 'Index of accessibility to radial highways',
    'TAX': 'Full-value property-tax rate per $10,000',
    'PTRATIO': 'Pupil-teacher ratio by town',
    'B': '1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',
    'LSTAT': 'Percentage of lower status of the population',
    'MEDV': 'Median value of owner-occupied homes in $1000s'
}

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Visualization", "🔮 Prediction", "🤖 Model Info"])

# Home Page
if page == "🏠 Home":
    st.title("Boston Housing Price Predictor")
    st.image("https://images.unsplash.com/photo-1565127803082-69dd82351360?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Ym9zdG9ufGVufDB8fDB8fHww", caption="Aerial view of Boston’s urban core — where history meets modern skyline.", use_container_width=True)

    st.markdown("""
    ## Welcome to the Boston Housing Price Prediction App
    
    This application predicts the median value of owner-occupied homes in Boston 
    based on various neighborhood characteristics using a machine learning model.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Key Features:**
        - Explore the Boston Housing dataset with interactive visualizations
        - Predict home values based on neighborhood features
        - Understand model performance and feature importance
        """)
    
    with col2:
        st.markdown("""
        **How to Use:**
        1. Navigate to different sections using the sidebar
        2. Adjust feature values in the Prediction section
        3. View model details and data visualizations
        """)
    
    st.markdown("---")
    st.subheader("About the Dataset")
    st.markdown("""
    The Boston Housing dataset contains information collected by the U.S Census Service 
    concerning housing in the area of Boston Mass. It has 506 entries with 14 features.
    """)



# Data Visualization Page
elif page == "📊 Data Visualization":
    st.title("Data Exploration")
    
    tab1, tab2, tab3 = st.tabs(["Descriptive Statistics", "Feature Distributions", "Relationships"])
    
    with tab1:
        st.subheader("Descriptive Statistics")
        st.dataframe(data.describe(), use_container_width=True)
        
    with tab2:
        st.subheader("Feature Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_feature = st.selectbox("Select feature to visualize", X.columns)
            
            st.markdown(f"**{selected_feature}**")
            st.markdown(f"*{feature_descriptions[selected_feature]}*")
            
            fig, ax = plt.subplots()
            sns.histplot(data[selected_feature], kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_feature}")
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(data=data[selected_feature], ax=ax)
            ax.set_title(f"Box Plot of {selected_feature}")
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Feature Relationships")
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis feature", X.columns, index=0)
        with col2:
            y_feature = st.selectbox("Y-axis feature", X.columns, index=5)
            
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=data, x=x_feature, y=y_feature, hue='MEDV', 
                       palette='viridis', ax=ax)
        ax.set_title(f"{x_feature} vs {y_feature} colored by MEDV")
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# Prediction Page
elif page == "🔮 Prediction":
    st.title("Price Prediction")
    
    st.markdown("""
    ### Predict Home Value
    
    Adjust the feature values below to get a prediction for the median home value.
    The model will estimate the value based on neighborhood characteristics.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Features")
        zn = st.slider(
            "Proportion of residential land zoned for large lots (ZN)",
            float(data['ZN'].min()), float(data['ZN'].max()), 
            float(data['ZN'].median())
        )
        indus = st.slider(
            "Proportion of non-retail business acres (INDUS)",
            float(data['INDUS'].min()), float(data['INDUS'].max()),
            float(data['INDUS'].median())
        )
        nox = st.slider(
            "Nitric oxides concentration (NOX)",
            float(data['NOX'].min()), float(data['NOX'].max()),
            float(data['NOX'].median())
        )
        
    with col2:
        st.subheader("Additional Features")
        age = st.slider(
            "Proportion of old owner-occupied units (AGE)",
            float(data['AGE'].min()), float(data['AGE'].max()),
            float(data['AGE'].median())
        )
        tax = st.slider(
            "Property tax rate per $10,000 (TAX)",
            float(data['TAX'].min()), float(data['TAX'].max()),
            float(data['TAX'].median())
        )
        ptratio = st.slider(
            "Pupil-teacher ratio (PTRATIO)",
            float(data['PTRATIO'].min()), float(data['PTRATIO'].max()),
            float(data['PTRATIO'].median())
        )
    
    # Set default values for other features
    default_values = {
        'CRIM': data['CRIM'].median(),
        'CHAS': 0,
        'RM': data['RM'].median(),
        'DIS': data['DIS'].median(),
        'RAD': data['RAD'].median(),
        'B': data['B'].median(),
        'LSTAT': data['LSTAT'].median()
    }
    
    if st.button("Predict Home Value", type="primary"):
        # input array in the correct feature order
        input_features = [
            default_values['CRIM'], zn, indus, default_values['CHAS'], nox,
            default_values['RM'], age, default_values['DIS'], default_values['RAD'],
            tax, ptratio, default_values['B'], default_values['LSTAT']
        ]
        
        # Make prediction
        prediction = model.predict(np.array(input_features).reshape(1, -1))[0]
        
        # Display result
        st.success(f"### Predicted Home Value: ${prediction * 1000:,.2f}")
        
        # Show feature importance
        st.markdown("---")
        st.subheader("Feature Impact on Prediction")
        
        coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        # Highlight the features the user adjusted
        adjusted_features = ['ZN', 'INDUS', 'NOX', 'AGE', 'TAX', 'PTRATIO']
        coefficients['Adjusted'] = coefficients['Feature'].isin(adjusted_features)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=coefficients, x='Coefficient', y='Feature', 
                   hue='Adjusted', dodge=False, palette={True: '#1f77b4', False: '#d3d3d3'})
        ax.set_title("Feature Coefficients (Impact on Home Value)")
        ax.axvline(0, color='black', linestyle='--')
        ax.legend_.remove()
        st.pyplot(fig)
        
        st.markdown("*Blue bars indicate features you adjusted in this prediction*")

# Model Info Page
elif page == "🤖 Model Info":
    st.title("Model Information")
    
    st.markdown("""
    ## Linear Regression Model
    
    This app uses a linear regression model trained on the Boston Housing dataset
    to predict median home values based on neighborhood characteristics.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
        st.markdown("""
        **R² Score Interpretation:**
        - 1.0: Perfect prediction
        - 0.0: No better than predicting the mean
        - Negative: Worse than predicting the mean
        """)
    
    with col2:
        st.metric("Mean Squared Error", f"{mse:.3f}")
        st.markdown("""
        **MSE Interpretation:**
        - Measures average squared difference between predicted and actual values
        - Lower values indicate better performance
        - In the same units as the target (MEDV) squared
        """)
    
    st.markdown("---")
    st.subheader("Model Coefficients")
    
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
        'Absolute Impact': abs(model.coef_)
    }).sort_values('Absolute Impact', ascending=False)
    
    st.dataframe(coefficients.drop(columns='Absolute Impact'), 
                use_container_width=True)
    
    st.markdown("""
    **Coefficient Interpretation:**
    - Positive: As feature increases, predicted home value increases
    - Negative: As feature increases, predicted home value decreases
    - Magnitude shows relative importance (when features are normalized)
    """)
    
    st.markdown("---")
    st.subheader("How the Prediction Works")
    st.markdown("""
    1. The model was trained on the entire Boston Housing dataset
    2. For predictions:
       - We take your input values for the selected features
       - Use median values for features you didn't adjust
       - Apply the trained model to generate a prediction
    3. The result is the predicted median home value in $1000s
    """)

# Add some custom styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stSuccess {
        background-color: #e6f7e6;
        border-radius: 5px;
        padding: 1rem;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)




