
# import matplotlib.pyplot as plt
# import requests as re
# import streamlit as st
# from bs4 import BeautifulSoup

# st.markdown(
#     """
#     <style>
#     .css-wjbhl0.e1fqkh3o9 {
#         display: none;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# def show():
#     st.markdown("<h1 style='color:#c8a808'>Project Details</h1>", unsafe_allow_html=True)
#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)

#     st.markdown("<h4 style='color:#4d6cc1'>Approach</h4>", unsafe_allow_html=True)
#     # st.subheader('Approach')
#     st.write('We used _supervised learning_ to classify phishing and legitimate websites. '
#              'We benefit from content-based approach and focus on html of the websites. '
#              'Also, We used scikit-learn for the ML models.'
#              )

#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#4d6cc1'>Conceptual Diagram</h4>", unsafe_allow_html=True)
#     # Load the image
#     phising_architect = "static/phising_architecture.jpg"

#     # Display the image
#     st.image(phising_architect, caption='Conceptual Diagram', use_column_width=True)

#     st.write('For this educational project, '
#              'We created my own data set and defined features, some from the literature and some based on manual analysis. '
#              'We used requests library to collect data, BeautifulSoup module to parse and extract features. ')
#     st.write('The source code and data sets are available in the below Github link:')
#     st.write('_https://github.com/AdarshVajpayee19/Phishing-Website-Detection-ML_')
#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#4d6cc1'>Data set </h4>", unsafe_allow_html=True)

#     st.write('We used _"phishtank.org"_ & _"tranco-list.eu"_ as data sources.')
#     st.write('Totally 26584 websites ==> **_16060_ legitimate** websites | **_10524_ phishing** websites')

#     # ----- FOR THE PIE CHART ----- #
#     labels = 'phishing', 'legitimate'
#     phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
#     legitimate_rate = 100 - phishing_rate
#     sizes = [phishing_rate, legitimate_rate]
#     explode = (0.1, 0)
#     fig, ax = plt.subplots()
#     ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
#     ax.axis('equal')
#     st.pyplot(fig)
#     # ----- !!!!! ----- #
#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.write('Features + URL + Label ==> Dataframe')
#     st.markdown('label is 1 for phishing, 0 for legitimate')
#     number = st.slider("Select row number to display", 0, 100)
#     st.dataframe(ml.legitimate_df.head(number))


#     @st.cache
#     def convert_df(df):
#         # IMPORTANT: Cache the conversion to prevent computation on every rerun
#         return df.to_csv().encode('utf-8')

#     csv = convert_df(ml.df)

#     st.download_button(
#         label="Download data as CSV",
#         data=csv,
#         file_name='phishing_legitimate_structured_data.csv',
#         mime='text/csv',
#     )
#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#4d6cc1'>Features</h4>", unsafe_allow_html=True)

#     st.write('We used only content-based features.We didn\'t use url-based faetures like length of url etc.'
#              'Most of the features extracted using find_all() method of BeautifulSoup module after parsing html.')

#     st.markdown("<h4 style='color:#4d6cc1'>Results</h4>", unsafe_allow_html=True)

#     st.write('We used 7 different ML classifiers of scikit-learn and tested them implementing k-fold cross validation.'
#              'Firstly obtained their confusion matrices, then calculated their accuracy, precision and recall scores.'
#              'Comparison table is below:')
#     st.table(ml.df_results)
#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.write('NB --> Gaussian Naive Bayes')
#     st.write('SVM --> Support Vector Machine')
#     st.write('DT --> Decision Tree')
#     st.write('RF --> Random Forest')
#     st.write('AB --> AdaBoost')
#     st.write('NN --> Neural Network')
#     st.write('KN --> K-Neighbours')
#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.subheader("Phishing Demo.")
#     # st.image("static\phising.gif", use_column_width=True)
#     # st.markdown(
#     #     '<img src="./app/static/phising.gif">',
#     #     unsafe_allow_html=True,
#     # )

#     # Load the GIF
#     phising = "static/phising.gif"

#     # Display the GIF
#     st.image(phising, caption='Animated Phishing Example GIF', use_column_width=True)
#     st.markdown("<hr>", unsafe_allow_html=True)





#     # Title and Description
#     st.title("End-to-End Machine Learning Project: Phishing Website Detection")
#     st.write("This is an End-to-End Machine Learning Project which focuses on phishing websites to classify phishing and legitimate ones. Particularly, We focused on content-based features like html tag based features. You can find feature extraction, data collection, preparation process here. Also, building ML models, evaluating them are available here.")

#     # Inputs
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#4d6cc1'>Inputs</h4>", unsafe_allow_html=True)

#     st.write("CSV files of phishing and legitimate URLs:")
#     st.write("- verified_online.csv: phishing websites URLs from phishtank.org")
#     st.write("- tranco_list.csv: legitimate websites URLs from tranco-list.eu")

#     # General Flow
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#4d6cc1'>General Flow</h4>", unsafe_allow_html=True)

#     st.write("1. Use csv file to get URLs")
#     st.write("2. Send a request to each URL and receive a response by requests library of python")
#     st.write("3. Use the content of response and parse it by BeautifulSoup module")
#     st.write("4. Extract features and create a vector which contains numerical values for each feature")
#     st.write("5. Repeat feature extraction process for all content\websites and create a structured dataframe")
#     st.write("6. Add label at the end to the dataframes | 1 for phishing 0 for legitimate")
#     st.write("7. Save the dataframe as csv and structured_data files are ready!")
#     st.write("8. Check 'structured_data_legitimate.csv' and 'structured_data_phishing.csv' files.")
#     st.write("9. After obtaining structured data, you can use combine them and use them as train and test data")
#     st.write("10. You can split data as train and test like in the machine_learning.py first part, or you can implement K-fold cross-validation like in the second part of the same file. We implemented K-fold as K=5.")
#     st.write("11. Then We implemented five different ML models:")
#     st.write("   - Support Vector Machine")
#     st.write("   - Gaussian Naive Bayes")
#     st.write("   - Decision Tree")
#     st.write("   - Random Forest")
#     st.write("   - AdaBoost")
#     st.write("12. You can obtain the confusion matrix, and performance measures: accuracy, precision, recall")
#     st.write("13. Finally, We visualized the performance measures for all models.")
#     st.write("14. Naive Bayes is the best for my case.")

#     # Important Notes
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#4d6cc1'>Important Notes</h4>", unsafe_allow_html=True)

#     st.write("Features are content-based and need BeautifulSoup module's methods and fields etc So, you should install it.")

#     # Dataset
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#4d6cc1'>Data set</h4>", unsafe_allow_html=True)

#     st.write("With your URL list, you can create your own dataset by using data_collector python file.")

#     st.markdown("<hr>", unsafe_allow_html=True)




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils import footer

st.markdown(
    """
    <style>
    .css-wjbhl0.e1fqkh3o9 {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
def show():
    st.markdown("<h1 style='color:#c8a808'>Project Details</h1>", unsafe_allow_html=True)
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)

    # 1. Loading the data
    st.markdown("<h3 style='color:#4d6cc1'>1. Loading the data</h3>", unsafe_allow_html=True)

    st.write("The dataset is borrowed from Kaggle, [Phishing Website Detector](https://www.kaggle.com/eswarchandt/phishing-website-detector).")
    st.write("A collection of website URLs for 11000+ websites. Each sample has 30 website parameters and a class label identifying it as a phishing website or not (1 or -1).")
    st.write("The overview of this dataset is, it has 11054 samples with 32 features.")


    # Loading data into dataframe
    data = pd.read_csv("phishing.csv")
    st.markdown("<h4 style='color:#4d6cc1'>Phishing Dataset Overview</h4>", unsafe_allow_html=True)

    # Display dataset with slider functionality
    number = st.slider("Select number of rows to display", 1, len(data), 5)
    st.write("Displaying first", number, "rows of the dataset:")
    st.write(data.head(number))

    # Function to convert DataFrame to CSV and cache the result
    @st.cache
    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')

    # Download button for the dataset
    csv = convert_df_to_csv(data)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing.csv',
        mime='text/csv',
    )

    # 2. Familiarizing with data & EDA
    st.markdown("<h3 style='color:#4d6cc1'>2. Familiarizing with data & EDA</h3>", unsafe_allow_html=True)
    st.write("Shape of dataframe:", data.shape)
    # Display features in a visually appealing way
    st.subheader("Features in the dataset:")
    features_list = ", ".join(data.columns.tolist())
    st.markdown(f"**Features:** {features_list}")
    st.subheader("Data Set OBSERVATIONS:")
    st.write("- There are 11054 instances and 31 features in dataset.")
    st.write("- Out of which 30 are independent features where as 1 is dependent feature.")
    st.write("- Each feature is in int datatype, so there is no need to use LabelEncoder.")
    st.write("- There is no outlier present in dataset.")
    st.write("- There is no missing value in dataset.")

    # 3. Visualizing the data
    st.markdown("<h3 style='color:#4d6cc1'>3. Visualizing the data</h3>", unsafe_allow_html=True)
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(15,15))
    sns.heatmap(data.corr(), annot=True)
    st.pyplot()

    # Pairplot
    st.markdown("<h3 style='color:#4d6cc1'>Pairplot for particular features</h3>", unsafe_allow_html=True)
    df = data[['PrefixSuffix-', 'SubDomains', 'HTTPS','AnchorURL','WebsiteTraffic','class']]
    sns.pairplot(data = df,hue="class",corner=True)
    st.pyplot()

    # Phishing Count in pie chart
    st.subheader("Phishing Count")
    phishing_count = data['class'].value_counts()
    plt.pie(phishing_count, labels=phishing_count.index, autopct='%1.2f%%')
    plt.title("Phishing Count")
    st.pyplot()

    # 4. Splitting the data
    st.markdown("<h3 style='color:#4d6cc1'>4. Splitting the data</h3>", unsafe_allow_html=True)
    st.write("The data is split into train & test sets, 80-20 split.")
    X = data.drop(["class"],axis =1)
    y = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    st.write("Train set shape:", X_train.shape, y_train.shape)
    st.write("Test set shape:", X_test.shape, y_test.shape)

    # 5. Model Building & Training

    st.markdown("<h3 style='color:#4d6cc1'>5. Model Building & Training</h3>", unsafe_allow_html=True)
    st.write("Supervised machine learning is one of the most commonly used and successful types of machine learning.")
    st.write("Supervised learning is used whenever we want to predict a certain outcome/label from a given set of features, and we have examples of features-label pairs.")
    st.write("The supervised machine learning models (regression) considered to train the dataset in this notebook are:")
    # Define model descriptions
    model_descriptions = {
        "Logistic Regression": "Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. Logistic Regression is much similar to the Linear Regression except that how they are used. Linear Regression is used for solving Regression problems, whereas Logistic regression is used for solving the classification problems.",

        "k-Nearest Neighbors": "K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique. K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.",

        "Support Vector Classifier": " Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future.",

        "Naive Bayes": "Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems.It is mainly used in text, image classification that includes a high-dimensional training dataset. Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.",

        "Decision Tree": "Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.",

        "Random Forest": " Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.",

        "Gradient Boosting": "Gradient boosting classifiers are a group of machine learning algorithms that combine many weak learning models together to create a strong predictive model. Decision trees are usually used when doing gradient boosting. Boosting algorithms play a crucial role in dealing with bias variance trade-off. Unlike bagging algorithms, which only controls for high variance in a model, boosting controls both the aspects (bias & variance), and is considered to be more effective.",

        "Catboost": "CatBoost is a recently open-sourced machine learning algorithm from Yandex. It can easily integrate with deep learning frameworks like Google’s TensorFlow and Apple’s Core ML. It can work with diverse data types to help solve a wide range of problems that businesses face today.",

        "Xgboost": "XGBoost is an implementation of gradient boosted decision trees designed for speed and performance that is dominative competitive machine learning.",

        "Multilayer Perceptrons": "MLPClassifier stands for Multi-layer Perceptron classifier which in the name itself connects to a Neural Network. Unlike other classification algorithms such as Support Vectors or Naive Bayes Classifier, MLPClassifier relies on an underlying Neural Network to perform the task of classification."
    }

    # Display model descriptions
    for model, description in model_descriptions.items():
        # st.markdown(f"## {model}")
        # st.write(description)
        with st.expander(f"## {model}"):
            st.write(description)


    # 6. Comparison of Model

    st.markdown("<h3 style='color:#4d6cc1'>6. Comparison of Model</h3>", unsafe_allow_html=True)
    st.write("To compare the models performance, a dataframe is created. The columns of this dataframe are the lists created to store the results of the model.")

    # Displaying the table
    st.markdown("""
    ## Result

    Accuracy of various model used for URL detection

    ||ML Model|	Accuracy|  	f1_score|	Recall|	Precision|
    |---|---|---|---|---|---|
    0|	Gradient Boosting Classifier|	0.974|	0.977|	0.994|	0.986|
    1|	CatBoost Classifier|	        0.972|	0.975|	0.994|	0.989|
    2|	XGBoost Classifier| 	        0.969|	0.973|	0.993|	0.984|
    3|	Multi-layer Perceptron|	        0.969|	0.973|	0.995|	0.981|
    4|	Random Forest|	                0.967|	0.971|	0.993|	0.990|
    5|	Support Vector Machine|	        0.964|	0.968|	0.980|	0.965|
    6|	Decision Tree|      	        0.960|	0.964|	0.991|	0.993|
    7|	K-Nearest Neighbors|        	0.956|	0.961|	0.991|	0.989|
    8|	Logistic Regression|        	0.934|	0.941|	0.943|	0.927|
    9|	Naive Bayes Classifier|     	0.605|	0.454|	0.292|	0.997|
    """)

    # 7. Storing Best Model

    st.markdown("<h3 style='color:#4d6cc1'>7. Storing Best Model</h3>", unsafe_allow_html=True)
    st.write("The best performing model is XGBoost Classifier")

    # 8. Conclusion
    st.markdown("<h3 style='color:#4d6cc1'>8. Conclusion</h3>", unsafe_allow_html=True)
    st.write("1. The final take away form this project is to explore various machine learning models, perform Exploratory Data Analysis on phishing dataset and understanding their features.")
    st.write("2. Creating this notebook helped me to learn a lot about the features affecting the models to detect whether URL is safe or not, also I came to know how to tuned model and how they affect the model performance.")
    st.write("3. The final conclusion on the Phishing dataset is that the some feature like 'HTTTPS', 'AnchorURL', 'WebsiteTraffic' have more importance to classify URL is phishing URL or not.")
    st.write("4. Gradient Boosting Classifier currectly classify URL upto 97.4% respective classes and hence reduces the chance of malicious attachments.")

    # 9. Display the best model

    st.markdown("<h3 style='color:#4d6cc1'>9. Display the best model</h3>", unsafe_allow_html=True)
    st.markdown("<h44>XGBoost Classifier Model</h44>", unsafe_allow_html=True)
    # XGBoost Classifier Model
    gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)
    gbc.fit(X_train,y_train)
    # Checking the feature importance in the model
    st.subheader("Feature Importance")
    plt.figure(figsize=(9,7))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), gbc.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.title("Feature importances using permutation on full model")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    st.pyplot()
    
    st.markdown("<h3 style='color:#4d6cc1'>10. Ensembling all the 10 Models into 1 Single model.</h3>", unsafe_allow_html=True)
    # Step 1: Importing Libraries
    st.markdown("<h5 style='color:#4d6cc1'> <b>Step 1:</b> Importing Libraries</h5>", unsafe_allow_html=True)
    st.write("In this step, we import the necessary libraries for data manipulation, visualization, and machine learning algorithms.")

    # Step 2: Loading Data
    st.markdown("<h5 style='color:#4d6cc1'> <b>Step 2:</b> Loading Data</h5>", unsafe_allow_html=True)
    st.write("Load your dataset into a pandas DataFrame using `pd.read_csv()`. Drop the index column if it's not necessary.")

    # Step 3: Splitting Data
    st.markdown("<h5 style='color:#4d6cc1'> <b>Step 3:</b> Splitting Data</h5>", unsafe_allow_html=True)
    st.write("Split the dataset into independent features (X) and the target variable (y). Then, further split the data into training and testing sets using `train_test_split()` from scikit-learn.")

    # Step 4: Model Training
    st.markdown("<h5 style='color:#4d6cc1'> <b>Step 4:</b> Model Training</h5>", unsafe_allow_html=True)
    st.write("Train several classifiers on the training data. These classifiers include Logistic Regression, K-Nearest Neighbors, Support Vector Classifier, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, CatBoost, XGBoost, and Multi-layer Perceptron Classifier.")

    # Step 5: Ensemble Model
    st.markdown("<h5 style='color:#4d6cc1'> <b>Step 5:</b> Ensemble Model</h5>", unsafe_allow_html=True)
    st.write("Create an ensemble model by combining the individual models. The ensemble model predicts the final class label by taking a majority vote from the predictions of each individual model.")

    # Step 6: Saving the Model
    st.markdown("<h5 style='color:#4d6cc1'> <b>Step 6:</b> Saving the Model</h5>", unsafe_allow_html=True)
    st.write("Finally, save the ensemble model into a file using the `pickle` library. This file can be later loaded and used for making predictions.")

    st.markdown("## Technologies Used")

    # Display badges
    st.markdown("![](https://forthebadge.com/images/badges/made-with-python.svg)")

    # Display logos with links
    st.markdown("[<img target='_blank' src='https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg' width=200>](https://numpy.org/doc/) "
                "[<img target='_blank' src='https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg' width=200>](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) "
                "[<img target='_blank' src='https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg' width=100>](https://matplotlib.org/) "
                "[<img target='_blank' src='https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png' width=200>](https://scikit-learn.org/stable/) ",
                unsafe_allow_html=True)
    # # Call the footer function to display the footer
    # footer()
