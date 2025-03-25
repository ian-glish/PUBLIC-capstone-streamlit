import streamlit as st 
import pandas as pd 
import plotly.express as px 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.preprocessing import StandardScaler

# Set page title and icon
st.set_page_config(page_title="Diabetes Health Indicators Classification Prediction Project", page_icon="🏥")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Training Models & Their Evaluations", "Make Your Own Predictions!"])

# Load dataset
#cleaned train dataset
df = pd.read_csv('data/cleaned_diabetes.csv')
# Original train dataset used for EDA for label purposes
ldf = pd.read_csv('data/labeled_diabetes_ds.csv')


# Home Page
if page == "Home":
    st.title("Predicting Diabetic Status Through Classification Machine Learning")
    st.subheader("By Ian Glish")
    st.write("""
        This Streamlit app provides an interactive platform to explore the diabetes health indicators dataset, sourced from Kaggle.
    """)
    st.write("""
        This app provides data overview information, exploratory data analysis, training 3 types of classification machine learning models with their accuracy evaluations and a logistic regression classification machine learning model that helps predict the diabetic status of a person based on 24 features through user input via sliding scales.
    """)
    st.write("""
        Logistic regression model type was chosen because of its highest accuracy score on new data, out of the 3 different types of classification models shown in this Streamlit app.
    """)
    st.image('images/DBinfographic.jpg')
    st.write("""
    Please be aware that some aspects of this app might be slow due to dataset size, so please be patient!
    """)


# Data Overview
elif page == "Data Overview":
    st.title("Data Overview")

    st.subheader("About the Data")
    st.write("""
      The diabetes health indicators dataset includes data was dervied from the 2015 CDC Behavioral Risk Factor Serveillance System survey, with information from over 70,692 respondents. Respondents were asked 21 different questions, to be used as features, regaurding to their current health status, life style choices and socioeconomic factors.
    """) 
    st.write("""
      The dataset was pre-numerically encoded, but contained both numeric and categorical features, including the target variable of whether a respondent had a diabetic status of either 'non-diabetic' or 'pre-diabetic or diabetic'.
    """)
    st.write("""
      3 additional engineered features were created from the 21 original features, creating a total of 24 features.
    """)
    st.write("This dataset was sourced from Kaggle: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    st.subheader("Data Dictionary")
    st.markdown("""
    | Feature   | Explination|
    | ----------- | ------------ |  
    |*Diabetic Status* | A respondent's diabetic status - Diabetic OR Prediabetic or Diabetic - Our target variable |        
    |*High Blood Pressure* | If a respondent has been told they have high blood pressure by a doctor, nurse, or other health professional: 0 - No, 1 - Yes| 
    |*High Cholesterol*| If a respondent has ever been told that their blood cholesterol is high by a doctor, nurse or other health professional: 0 - No, 1 - Yes |      
    |*Cholesterol Check*| If a respondent has had a cholesterol check in the last 5 years: 0 - No, 1 - Yes|          
    |*BMI*| Body Mass Index: Numeric value representing a respondent's BMI |           
    |*Smoker*| If a respondent has smoked at least 100 cigarettes in their entire life: 0 - No, 1 - Yes|          
    |*Stroke*| If a respondent has ever been told that they had a stroke: 0 - No, 1 - Yes|           
    |*Heart Disease or Attack*| If a respondent has ever reported having a coronary heart disease or myocardial infarction: 0 - No, 1 - Yes|          
    |*Physical Activity*| If a respondent reported doing physical activity or exercise during the past 30 days other than their regular job: 0 - No, 1 - Yes|            
    |*Consumes Fruit*|If a respondent consumes fruit one or more times a day: 0 - No, 1 - Yes|            
    |*Consumes Vegetables*|If a respondent consumes vegetables one or more times a day: 0 - No, 1 - Yes|         
    |*Heavy Alcohol Consumption*| For heavy alcohol consumers: if adult men have more than 14 drinks per week or adult women having more than 7 drinks per week: 0 - No, 1 - Yes |          
    |*Health Care*| If a respondent has any kind of health care coverage, including health insurance, prepaid plans such as HMOs or government plans such as Medicare or Indian Health Service: 0 - No, 1 - Yes |          
    |*No Doctor Due To Cost*| If there was a time in the past 12 months that a respondent needed to see a doctor but couldn't because of cost: 0 - No, 1 - Yes|     
    |*General Health*| Asking the respondent what they would rate their own health on a scale of 1-5: 1 - Poor, 2 - Fair, 3 - Good, 4 - Very Good, 5 - Excellent|          
    |*Mental Health*| Asking the respondent how many days in the last 30 days was their mental health not good, in regards to stress, depression or emotional problems: Range of 0 - 30|          
    |*Physical Health*| Asking the respondent how many days in the last 30 days was their physical health not good - in regards to physically illness and injury: Range of 0 - 30 |         
    |*Difficulty Walking*| If a respondent has serious difficulty walking or climbing stairs: 0 - No, 1 - Yes|      
    |*Sex*| A respondent's gender: 0 - Female, 1 - Male|         
    |*Age*| An age classification code based on the respondent's age: 1: 18-24, 2: 25-29, 3: 30-34, 4:35-39, 5:40-44, 6:45-49, 7:50-54, 8:55-59, 9:60-64, 10:65-69, 11:70-74, 12:75-79, 13:80 Or Older |         
    |*Education*| A education classification code based on the respondent's education level: 1:No School, 2:Elementary, 3:Some High School, 4:High School Graduate, 5:Some College, 6:College Graduate|         
    |*Income*| An income classification code base on the respondent's income level: 1:Less Than \$10K , 2:Less Than \$15K, 3:Less Than \$20K, 4:Less Than \$25K, 5:Less Than \$35K, 6:Less Than \$50K, 7:Less Than \$75K, 8:\$75K Or More  |      
    |*Produce Consumption Score*| A self created engineered feature that combines a respondent's fruit and vegetable consumption scores, for a maximum out of 2. This feature adds up a respondent's 'Consumes Fruit' & 'Consumes Vegetables' values together |
    |*Overall Health Score*| A self created engineered feature that combines a respondent's scores for their mental and physical health in the last 30 days, for a total maximum of 60. This feature adds up a respondent's 'Mental Health' & 'Physical Health' self ratings together| 
    """)

    # Dataset Display
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    

    # Shape of Dataset
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis\nUsing Plotly Visualizations")


    container = st.container(border=True)
    container.subheader("Select the type of visualization you'd like to explore:")
    eda_type = container.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])

    obj_cols = ldf.select_dtypes(include='object').columns.tolist()
    num_cols = ldf.select_dtypes(include='number').columns.tolist()

    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numeric variable for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.replace('_', ' ')}"
            if st.checkbox("Show by Diabetic Status"):
                diabetes_color_change = {'Non-Diabetic':'yellow', 'Pre-Diabetic or Diabetic':'green'}
                st.plotly_chart(px.histogram(ldf, x=h_selected_col, color = 'Diabetic Status', color_discrete_map = diabetes_color_change, title=chart_title, barmode='overlay', opacity=.8))
            else:
                st.plotly_chart(px.histogram(ldf, x=h_selected_col, title=chart_title))
                

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numeric variable for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by Diabetic Status"):
                diabetes_color_change = {'Non-Diabetic':'yellow', 'Pre-Diabetic or Diabetic':'red'}
                st.plotly_chart(px.box(ldf, x=b_selected_col, color='Diabetic Status', color_discrete_map = diabetes_color_change, title=chart_title))
            else:
                st.plotly_chart(px.box(ldf, x=b_selected_col, title=chart_title))

    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols)
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            if st.checkbox("Show by Diabetic Status"):
                diabetes_color_change = {'Non-Diabetic':'yellow', 'Pre-Diabetic or Diabetic':'red'}
                st.plotly_chart(px.scatter(ldf, x=selected_col_x, y=selected_col_y, color='Diabetic Status', color_discrete_map = diabetes_color_change, title=chart_title))
            else:
                st.plotly_chart(px.scatter(ldf, x=selected_col_x, y=selected_col_y, title=chart_title))

    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox("Select a categorical feature:", obj_cols)
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            if st.checkbox("Show by Diabetic Status"):
                diabetes_color_change = {'Non-Diabetic':'yellow', 'Pre-Diabetic or Diabetic':'red'}
                st.plotly_chart(px.histogram(ldf, x=selected_col, color='Diabetic Status', color_discrete_map = diabetes_color_change, title=chart_title, barmode='overlay', opacity = .8))

            else:
                st.plotly_chart(px.histogram(ldf, x=selected_col, title=chart_title))
        


# Model Training and Evaluation Page
elif page == "Training Models & Their Evaluations":
    st.title("Classification Model Types & Trainings with Performance Evaluations")
    st.subheader("Choose a classification model type to train on the dataset, to see its accuracy scores & corresponding confusion matrix.")
    st.write("The baseline model had an accuracy score of 50.00%, so any of these model types need to have a higher accuracy score than the baseline to be worth using!")


    # Sidebar for model selections
    st.sidebar.subheader("Choose a Classification Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a Model Type", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns = 'Diabetic Status')
    y = df['Diabetic Status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select The Number of Neighbors (K)", min_value=1, max_value=79, value=41)
        model = KNeighborsClassifier(n_neighbors=k, metric = 'euclidean')
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    elif model_option == "Random Forest":
        model = RandomForestClassifier(max_depth = 6)

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    container = st.container(border=True)
    container.write(f" **Model Selected: {model_option}**")
    container.write(f" **Training Accuracy: {model.score(X_train_scaled, y_train)*100:.2f}%**")
    container.write(f" **Test Accuracy: {model.score(X_test_scaled, y_test)*100:.2f}%**")

    # Display confusion matrix
    st.subheader(f"Confusion Matrix for {model_option} Model")
    fig, ax = plt.subplots()
    if model_option == "K-Nearest Neighbors":
        ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
        st.pyplot(fig)
        st.write(f"Using the {model_option} model, the confusion matrix shows a higher rate of misclassifying 'Non-Diabetic' as 'neutral or Pre-Diabetic or Diabetic' (false positives) than misclassifying 'Pre-Diabetic or Diabetic' as 'Non-Diabetic' (false negatives).")
        st.write(f"This suggests the {model_option} model has a strong tendency to label non-diabetics as pre-diabetic or diabetic, showing a weakness in producing false positive labels.")
        st.write(f"**Out of all the model types, {model_option} had the 3rd highest test set accuracy score, even with the best K value of 45.**")
    elif model_option == "Logistic Regression":
        ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Greens')
        st.pyplot(fig)
        st.write(f"Using the {model_option} model, the confusion matrix shows a higher rate of misclassifying 'Non-Diabetic' as 'neutral or Pre-Diabetic or Diabetic' (false positives) than misclassifying 'Pre-Diabetic or Diabetic' as 'Non-Diabetic' (false negatives).")
        st.write(f"This suggests the {model_option} model has a strong tendency to label non-diabetics as pre-diabetic or diabetic, showing a weakness in producing false positive labels.")
        st.write(f"**Out of all the model types, {model_option} had the highest test set accuracy score. This logistic regression model will be used on the 'Make Your Own Predictions!' page.**")
    elif model_option == "Random Forest":
        ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Oranges')
        st.pyplot(fig)
        st.write(f"Using the {model_option} model, the confusion matrix shows a higher rate of misclassifying 'Non-Diabetic' as 'neutral or Pre-Diabetic or Diabetic' (false positives) than misclassifying 'Pre-Diabetic or Diabetic' as 'Non-Diabetic' (false negatives).")
        st.write(f"This suggests the {model_option} model has a strong tendency to label non-diabetics as pre-diabetic or diabetic, showing a weakness in producing false positive labels.")
        st.write(f"**Out of all the model types, {model_option} had the second highest test accuracy score.**")
    # Make Predictions Page
elif page == "Make Your Own Predictions!":
    st.title("Make Your Own Diabetic Status Prediction")
    container = st.container(border=True)
    container.subheader("Use 23 features to input in a logistic regression classification model")
    container.subheader("**Adjust the feature scale values below to make your own predictions on whether someone would be classified as 'Non-Diabetic' or 'Pre-Diabetic or Diabetic'**")
    

    # User inputs for prediction
    high_blood_pressure = st.slider("High Blood Pressure: If told of high blood pressure by health professional: 0 - No, 1 - Yes", min_value=0, max_value=1, value=0)
    high_cholesterol = st.slider("High Cholesterol: If ever told by health professional of high cholesterol: 0 - No, 1 - Yes", min_value=0, max_value=1, value=0)
    cholesterol_check = st.slider("Cholesterol Check: If cholesterol has been checked in last 5 years: 0 - No, 1 - Yes", min_value=0, max_value=1, value=0)
    bmi = st.slider("BMI: Body Mass Index value:", min_value=10, max_value=100, value=10)
    smoker = st.slider("Smoker: If smoked 100 cigarettes in entire life: 0 - No, 1 - Yes", min_value=0, max_value=1, value=0)
    stroke = st.slider("Stroke: If ever had a stroke: 0 - No, 1 - Yes", min_value=0, max_value=1, value= 0)
    hdoa = st.slider("Heart Disease or Attack: If ever had coronary heart diesase or myocardial infarction: 0 - No, 1 - Yes:", min_value=0, max_value=1, value=0)
    physical_activity = st.slider("Physical Activity: Have done in last 30 days other than regular job: No, 1 - Yes:", min_value=0, max_value=1, value=0)
    consumes_fruit = st.slider("Consumes Fruit: Consumes fruit one or more times a day :", min_value=0, max_value=1, value=0)
    consumes_vegetables = st.slider("Consumes Vegetables: Consumes vegetables one or more times a day: 0 - No, 1 - Yes:", min_value=0, max_value=1, value=0)
    heavy_alcohol_consumption = st.slider("Heavy Alcohol Consumption: Men having < 14 drinks/week, Women having < 7 drinks/week: No, 1 - Yes", min_value=0, max_value=1, value=0)
    health_care = st.slider("Health Care: Having any kind of health care coverage: 0 - No, 1 - Yes", min_value=0, max_value=1, value=0)
    nddtc = st.slider("No Doctor Due To Cost: In last 12 months, no doctors visits due to cost: 0 - No, 1 - Yes", min_value=0, max_value=1, value=0)
    general_health = st.slider("General Health: General health overall rating: 1 - Poor, 2 - Fair, 3 - Good, 4 - Very Good, 5 - Excellent", min_value=1, max_value=5, value=1)
    mental_health = st.slider("Mental Health: How many days in last 30 days was mental health not good?:", min_value=0, max_value=30, value=0)
    physical_health = st.slider("Physical Health: How many days in the last 30 days was physical health not good?:", min_value=0, max_value=30, value=0)
    difficulty_walking = st.slider("Difficulty Walking: Any serious difficulty walking or climbing stairs?: 0 - No, 1 - Yes", min_value=0, max_value=1, value=0)
    sex = st.slider("Sex:  0 - Female, 1 - Male", min_value=0, max_value=1, value=1)
    age = st.slider("Age: Pick a numeric level based on age: 1 = 18-24, 2 = 25-29, 3 = 30-34, 4 = 35-39, 5 = 40-44, 6 = 45-49, 7 = 50-54, 8 = 55-59, 9 = 60-64, 10 = 65-69, 11 = 70-74, 12 = 75-79, 13 = 80 Or Older", min_value=1, max_value=13, value=1)
    education = st.slider("Education: What is the highest education level finished?: 1 = No School, 2 = Elementary, 3 = Some High School, 4 = High School Graduate, 5 = Some College, 6 = College Graduate:", min_value=1, max_value=6, value=1)
    income = st.slider("Income: Choose an numeric value based on income level: 1 = Less Than \$10K , 2 = Less Than \$15K, 3 = Less Than \$20K, 4 = Less Than \$25K, 5 = Less Than \$35K, 6 = Less Than \$50K, 7 = Less Than \$75K, 8 = \$75K Or More ", min_value=1, max_value=8, value=1)
    pcs = st.slider("Produce Consumption Score: The added combination of fruit & vegetable servings eaten a day: 0 to 2", min_value=0, max_value=2, value=0)
    ohs = st.slider("Overall Health Score: The added combination of number of days mental & physical health was not good: 0 to 60:", min_value=0, max_value=60, value=0)
    
    # User input dataframe
    user_input = pd.DataFrame({
        'High Blood Pressure': [high_blood_pressure],
        'High Cholesterol': [high_cholesterol],
        'Cholesterol Check': [cholesterol_check],
        'BMI': [bmi],
        'Smoker': [smoker],
        'Stroke': [stroke],
        'Heart Disease or Attack': [hdoa],
        'Physical Activity': [physical_activity],
        'Consumes Fruit': [consumes_fruit],
        'Consumes Vegetables': [consumes_vegetables],
        'Heavy Alcohol Consumption': [heavy_alcohol_consumption],
        'Health Care': [health_care],
        'No Doctor Due To Cost': [nddtc],
        'General Health': [general_health],
        'Mental Health': [mental_health],
        'Physical Health': [physical_health],
        'Difficulty Walking': [difficulty_walking],
        'Sex': [sex],
        'Age': [age],
        'Education': [education],
        'Income': [income],
        'Produce Consumption Score': [pcs],
        'Overall Health Score': [ohs]
    })

    st.write("### Your Input Values:")
    st.dataframe(user_input)

    # Using Logistic Regression model for predictions since this was the most accurate in terms of understanding the training and test data:
    model = LogisticRegression()
    X = df.drop(columns = 'Diabetic Status')
    y = df['Diabetic Status']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Train the model on the scaled data
    model.fit(X_scaled, y)

    # Make predictions
    prediction = model.predict(user_input_scaled)[0]


    # Display the result
    st.write(" ### Based on your input features, the model predicts that the diabetic status is:")
    st.write(f"# {prediction}")