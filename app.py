# # # #importing required libraries

# # # import pickle
# # # import warnings
# # # from urllib.parse import quote_plus

# # # import numpy as np
# # # import pandas as pd
# # # from flask import Flask, render_template, request
# # # from sklearn import metrics

# # # warnings.filterwarnings('ignore')
# # # from feature import FeatureExtraction

# # # file = open("pickle/model.pkl","rb")
# # # gbc = pickle.load(file)
# # # file.close()


# # # app = Flask(__name__)

# # # @app.route("/", methods=["GET", "POST"])
# # # def index():
# # #     if request.method == "POST":

# # #         url = request.form["url"]
# # #         obj = FeatureExtraction(url)
# # #         x = np.array(obj.getFeaturesList()).reshape(1,30)

# # #         y_pred =gbc.predict(x)[0]
# # #         #1 is safe
# # #         #-1 is unsafe
# # #         y_pro_phishing = gbc.predict_proba(x)[0,0]
# # #         y_pro_non_phishing = gbc.predict_proba(x)[0,1]
# # #         # if(y_pred ==1 ):
# # #         pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
# # #         return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
# # #     return render_template("index.html", xx =-1)


# # # if __name__ == "__main__":
# # #     app.run(debug=True)





# # import pickle
# # import warnings

# # import numpy as np
# # import pandas as pd
# # import streamlit as st

# # from feature import FeatureExtraction

# # warnings.filterwarnings('ignore')

# # # Load the trained model
# # file = open("pickle/model.pkl", "rb")
# # gbc = pickle.load(file)
# # file.close()

# # # Define the main function
# # def main():
# #     st.title("Phishing URL Detector")

# #     # Get URL input from the user
# #     url = st.text_input("Enter the URL:")
    
# #     # Add a button to initiate the URL check
# #     if st.button("Check URL"):
# #         if url:
# #             obj = FeatureExtraction(url)
# #             x = np.array(obj.getFeaturesList()).reshape(1, 30)

# #             # Make predictions
# #             y_pred = gbc.predict(x)[0]
# #             y_pro_phishing = gbc.predict_proba(x)[0, 0]
# #             y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

# #             # Display the result
# #             if y_pred == 1:
# #                 st.success("It is {:.2f}% safe to go".format(y_pro_non_phishing * 100))
# #             else:
# #                 st.error("It is {:.2f}% unsafe to go".format(y_pro_phishing * 100))

# # # Run the main function
# # if __name__ == "__main__":
# #     main()









# # import pickle
# # import warnings
# # from pathlib import Path

# # from feature import FeatureExtraction

# # warnings.filterwarnings('ignore')
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import requests as re
# # import streamlit as st
# # from bs4 import BeautifulSoup

# # # Import the FeatureExtraction class from feature.py
# # from feature import FeatureExtraction

# # st.set_page_config(page_title='Phishing Website Detection Using Machine Learning', page_icon='./static/favicon.png')
# # # Add the CSS rule using st.markdown
# # st.markdown(
# #     """
# #     <style>
# #     .css-wjbhl0.e1fqkh3o9 {
# #         display: none;
# #     }
# #     </style>
# #     """,
# #     unsafe_allow_html=True
# # )
# # # --- PATH SETTINGS ---
# # current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
# # css_file = current_dir / "styles" / "main.css"
# # phishing_account_pic = current_dir / "static" / "Phishing-account.gif"

# # # Load the trained model
# # file = open("pickle/model.pkl", "rb")
# # gbc = pickle.load(file)
# # file.close()


# # def applicationRun():

# #     # Add content for the Home page here
# #     # Set page title and description
# #     st.markdown("<h1 style='color:#c8a808'>Phishr</h1>", unsafe_allow_html=True)
# #     st.markdown("<h3 style='color:#4d6cc1'>Phish the Phisher before they phish you!!!</h3>", unsafe_allow_html=True)

# #     # Add a horizontal line
# #     st.markdown("<hr>", unsafe_allow_html=True)
# #     st.markdown("<h4 style='color:#4d6cc1'>Understanding Phishing Attack</h4>", unsafe_allow_html=True)
# #     st.write('Phishing attacks are a common type of cyber attack where malicious actors attempt to deceive individuals or organizations into revealing '
# #              'sensitive information such as usernames, passwords, credit card numbers, or other personal or financial data. These attacks typically '
# #              'involve impersonating a trusted entity, such as a bank, a government agency, a company, or even a colleague or friend.')

# #     # Load the GIF
# #     phishing_acc = "static/Phishing-account.gif"

# #     # Display the GIF
# #     st.image(phishing_acc, caption='PHISHr', use_column_width=True)

# #     st.markdown("<hr>", unsafe_allow_html=True)
# #     with st.expander('EXAMPLE PHISHING URLs:'):
# #         st.write('_https://rtyu38.godaddysites.com/_')
# #         st.write('_https://karafuru.invite-mint.com/_')
# #         st.write('_https://defi-ned.top/h5/#/_')
# #         st.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE! SO, THE EXAMPLES SHOULD BE UPDATED!')

# #     # Add a horizontal line
# #     st.markdown("<hr>", unsafe_allow_html=True)

# #     url = st.text_input('Enter the URL', key='url_input')
# #     if st.button("Check URL"):
# #         if url:
# #             obj = FeatureExtraction(url)
# #             x = np.array(obj.getFeaturesList()).reshape(1, 30)

# #             # Make predictions
# #             y_pred = gbc.predict(x)[0]
# #             y_pro_phishing = gbc.predict_proba(x)[0, 0]
# #             y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
            
# #             # Display the result
# #             if y_pred == 1:
# #                 st.success("It is {:.2f}% safe to go".format(y_pro_non_phishing * 100))
# #             else:
# #                 st.error("It is {:.2f}% unsafe to go".format(y_pro_phishing * 100))

# #     st.markdown("<hr>", unsafe_allow_html=True)
# #     st.markdown("<h4 style='color:#4d6cc1'>Mitigating Phishing Risks</h4>", unsafe_allow_html=True)

# #     st.write(
# #         'Phishing attacks pose significant risks to individuals, businesses, and organizations. They can lead to identity theft, financial loss, data breaches, '
# #         'and reputational damage. To protect against phishing attacks, it\'s essential to stay vigilant, be cautious of unsolicited emails or messages, '
# #         'verify the authenticity of websites and communications, and regularly update security measures such as antivirus software and firewalls. '
# #         'Additionally, education and awareness training for employees and users are crucial in preventing successful phishing attacks.')
# #     # Add a horizontal line
# #     st.markdown("<hr>", unsafe_allow_html=True)

# # from menu import streamlit_menu
# # from pages import FAQ, Blog, Contact_Us, Project_Description

# # selected = streamlit_menu()

# # if selected == "Home":
# #     applicationRun()
# # if selected == "Project_Description":
# #     Project_Description.show()
# # elif selected == "Contact_Us":
# #     Contact_Us.show()
# # elif selected == "FAQ":
# #     FAQ.show()
# # elif selected == "Blog":
# #     Blog.show()


# import pickle
# import warnings
# from pathlib import Path

# from feature import FeatureExtraction

# warnings.filterwarnings('ignore')
# import matplotlib.pyplot as plt
# import numpy as np
# import requests as re
# import streamlit as st
# from bs4 import BeautifulSoup

# # Import the FeatureExtraction class from feature.py
# from feature import FeatureExtraction
# from menu import streamlit_menu
# from pages import FAQ, Contact_Us, Project_Description
# from utils import footer

# # Set page config
# st.set_page_config(page_title='Phishing Website Detection Using Machine Learning', page_icon='./static/favicon.png')
# # Add the CSS rule using st.markdown
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
# # --- PATH SETTINGS ---
# current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
# css_file = current_dir / "styles" / "main.css"
# phishing_account_pic = current_dir / "static" / "Phishing-account.gif"


# import numpy as np
# import streamlit as st
# from sklearn.base import BaseEstimator, ClassifierMixin

# # def render_carousel():
# #  # Define the HTML code for the carousel
# #     html_code = """
# #     <!doctype html>
# #     <html lang="en">
# #     <head>
# #         <meta charset="utf-8">
# #         <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
# #         <style>
# #         /* The typing effect */
# #         @keyframes typing {
# #             from { width: 0 }
# #             to { width: 100% }
# #         }

# #         /* The typewriter cursor effect */
# #         @keyframes blink-caret {
# #             from, to { border-color: transparent }
# #             50% { border-color: #4d6cc1 }
# #         }

# #         /* The container */
# #         .typewriter-container {
# #             overflow: hidden; /* Ensures the text is not revealed until the animation */
# #             border-right: .15em solid #4d6cc1; /* The typewriter cursor */
# #             white-space: nowrap; /* Keeps the content on a single line */
# #             margin: 0 auto; /* Gives that scrolling effect as the typing happens */
# #             letter-spacing: .15em; /* Adjust as needed */
# #             animation:
# #                 typing 3.5s steps(40, end),
# #                 blink-caret .75s step-end infinite;
# #         }
# #     </style>
# #     <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
# #     <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
# #     <script src="https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
# #     <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
# #     <script>
# #             $(document).ready(function(){
# #                 $('.carousel').carousel({
# #                     interval: 2000
# #                 });

# #                 $('#carouselExampleIndicators').on('slide.bs.carousel', function (e) {
# #                     console.log('Slide event occurred:', e);
# #                 });

# #                 $('#carouselExampleIndicators').on('slid.bs.carousel', function (e) {
# #                     console.log('Slid event occurred:', e);
# #                 });
# #             });
# #         </script>
# #     </head>
# #     <body>
# #         <div class="typewriter-container">
# #             <h3 style='color:#4d6cc1'>Phish the Phisher before they phish you!!!</h3>
# #         </div>
# #         <div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel">
# #             <ol class="carousel-indicators">
# #             <li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>
# #             <li data-target="#carouselExampleIndicators" data-slide-to="1"></li>
# #             <li data-target="#carouselExampleIndicators" data-slide-to="2"></li>
# #             </ol>
# #             <div class="carousel-inner">
# #             <div class="carousel-item active">
# #                 <img class="d-block w-100" src="https://cdn.pixabay.com/photo/2023/05/26/00/23/hacker-8018467_1280.png" alt="First slide">
# #             </div>
# #             <div class="carousel-item">
# #                 <img class="d-block w-100" src="https://cdn.pixabay.com/photo/2023/09/18/07/07/ai-generated-8259795_1280.jpg" alt="Second slide">
# #             </div>
# #             <div class="carousel-item">
# #                 <img class="d-block w-100" src="https://cdn.pixabay.com/photo/2023/05/26/00/55/hacker-8018499_1280.png" alt="Third slide">
# #             </div>
# #             </div>
# #             <a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">
# #             <span class="carousel-control-prev-icon" aria-hidden="true"></span>
# #             <span class="sr-only">Previous</span>
# #             </a>
# #             <a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">
# #             <span class="carousel-control-next-icon" aria-hidden="true"></span>
# #             <span class="sr-only">Next</span>
# #             </a>
# #         </div>
         
        
# #     </body>
# #     </html>
# #     """
# #      # Render the HTML code in Streamlit
# #     st.markdown(html_code, unsafe_allow_html=True)

# def typeWriter():
#     typeWriter = """
#      <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>Bootstrap Carousel</title>
#         <style>
#          /* The typing effect */
#          @keyframes typing {
#              from { width: 0 }
#              to { width: 100% }
#          }
#          /* The typewriter cursor effect */
#          @keyframes blink-caret {
#              from, to { border-color: transparent }
#              50% { border-color: #4d6cc1 }
#          }
#          /* The container */
#          .typewriter-container {
#              overflow: hidden; /* Ensures the text is not revealed until the animation */
#              border-right: .15em solid #4d6cc1; /* The typewriter cursor */
#              white-space: nowrap; /* Keeps the content on a single line */
#              margin: 0 auto; /* Gives that scrolling effect as the typing happens */
#              letter-spacing: .15em; /* Adjust as needed */
#              animation:
#                  typing 3.5s steps(40, end),
#                  blink-caret .75s step-end infinite;
#          }
#      </style>
#         <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css">
#     </head>
#     <body>
#         <div class="typewriter-container">
#              <h3 style='color:#4d6cc1'>Phish the Phisher before they phish you!!!</h3>
#          </div>
#     </body>
#     </html>
#     """
#     st.markdown(typeWriter, unsafe_allow_html=True)


# def render_bootstrap_carousel():
#     # Define the HTML code for the carousel
#     html_code = """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>Bootstrap Carousel</title>
#         <style>
#          .carousel-container {
#              max-width: 850px;
#              margin: 0 auto;
#              height: 450px; /* Set a fixed height for the carousel */
#              overflow: hidden; /* Hide overflow content */
#          }
#          .carousel-inner img {
#              max-height: 100%; /* Ensure images don't exceed the height of the container */
#              width: auto; /* Maintain aspect ratio */
#              margin: 0 auto; /* Center images horizontally */
#          }
#      </style>
#         <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css">
#     </head>
#     <body>
#         <div class="container">
#             <!-- Carousel -->
#             <div id="myCarousel" class="carousel slide" data-ride="carousel">
#                 <!-- Indicators -->
#                 <ol class="carousel-indicators">
#                     <li data-target="#myCarousel" data-slide-to="0" class="active"></li>
#                     <li data-target="#myCarousel" data-slide-to="1"></li>
#                     <li data-target="#myCarousel" data-slide-to="2"></li>
#                 </ol>

#                 <!-- Slides -->
#                 <div class="carousel-inner">
#                     <div class="carousel-item active">
#                         <img class="d-block w-100" src="https://cdn.pixabay.com/photo/2023/05/26/00/23/hacker-8018467_1280.png" alt="First slide">
#                     </div>
#                     <div class="carousel-item">
#                         <img class="d-block w-100" src="https://cdn.pixabay.com/photo/2023/09/18/07/07/ai-generated-8259795_1280.jpg" alt="Second slide">
#                     </div>
#                     <div class="carousel-item">
#                         <img class="d-block w-100" src="https://cdn.pixabay.com/photo/2023/05/26/00/55/hacker-8018499_1280.png" alt="Third slide">
#                     </div>
#                 </div>

#                 <!-- Controls -->
#                 <a class="carousel-control-prev" href="#myCarousel" role="button" data-slide="prev">
#                     <span class="carousel-control-prev-icon" aria-hidden="true"></span>
#                     <span class="sr-only">Previous</span>
#                 </a>
#                 <a class="carousel-control-next" href="#myCarousel" role="button" data-slide="next">
#                     <span class="carousel-control-next-icon" aria-hidden="true"></span>
#                     <span class="sr-only">Next</span>
#                 </a>
#             </div>
#         </div>

#         <!-- JavaScript -->
#         <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js"></script>
#         <script src="https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js"></script>
#         <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js"></script>

#         <script>
#             // Initialize the carousel with an interval of 2 seconds
#             $('#myCarousel').carousel({
#                 interval: 2000
#             });

#             // Hook into the slide event
#             $('#myCarousel').on('slide.bs.carousel', function (e) {
#                 // Do something when the carousel slide event occurs
#                 console.log('Slide event occurred:', e);
#             });

#             // Hook into the slid event
#             $('#myCarousel').on('slid.bs.carousel', function (e) {
#                 // Do something when the carousel slid event occurs
#                 console.log('Slid event occurred:', e);
#             });
#         </script>
#     </body>
#     </html>
#     """
#     # Render the HTML code in Streamlit
#     st.components.v1.html(html_code, width=600, height=330)
    

# class EnsembleModel(BaseEstimator, ClassifierMixin):
#     def __init__(self, models):
#         self.models = models

#     def fit(self, X, y):
#         for model in self.models:
#             model.fit(X, y)
#         return self

#     def predict(self, X):
#         predictions = [model.predict(X) for model in self.models]
#         # Combine predictions (e.g., by averaging or voting)
#         # For simplicity, let's use majority voting
#         return np.mean(predictions, axis=0) > 0.5  # Use threshold as needed


# # Load the trained model
# file = open("pickle/ensemble_model.pkl", "rb")
# ensemble_model = pickle.load(file)
# file.close()


# def applicationRun():

#     st.markdown("<h1 style='color:#c8a808'>Phishr</h1>", unsafe_allow_html=True)
#     st.markdown("<hr>", unsafe_allow_html=True)

#     typeWriter()

#     # Call the function to render the carousel
#     render_bootstrap_carousel()
#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#4d6cc1'>Understanding Phishing Attack</h4>", unsafe_allow_html=True)
#     st.write('Phishing attacks are a common type of cyber attack where malicious actors attempt to deceive individuals or organizations into revealing '
#              'sensitive information such as usernames, passwords, credit card numbers, or other personal or financial data. These attacks typically '
#              'involve impersonating a trusted entity, such as a bank, a government agency, a company, or even a colleague or friend.')

#     # Load the GIF
#     phishing_acc = "static/Phishing-account.gif"

#     # Display the GIF
#     st.image(phishing_acc, caption='PHISHr', use_column_width=True)

#     st.markdown("<hr>", unsafe_allow_html=True)
#     with st.expander('EXAMPLE LEGITIMATE URLs:'):
#         st.write('https://www.twitter.com')
#         st.write('https://www.youtube.com')
#         st.write('https://www.google.com')
#         st.write('https://www.facebook.com')
#         st.write('https://www.instagram.com')
#         st.write('https://www.icicibank.com')
#         st.write('https://graphicriver.net')
#         st.write('https://www.sourceforge.net')
#         st.write('https://www.metro.co.uk')
#         st.write('https://www.stackoverflow.com')
#         st.write('https://www.olx.co.id')
#         st.write('https://www.indianexpress.com')
#         st.write('https://www.kenh14.vn')
#         st.write('https://www.otomoto.pl')
#         st.write('https://www.askubuntu.com')
#         st.caption('This links can be used !!!')

#     with st.expander('EXAMPLE PHISHING URLs:'):
#         st.write('https://www.kinemind.com')
#         st.write('https://www.eevee.tv')
#         st.write('https://www.grandcup.xyz')
#         st.write('https://www.hamt.jp')
#         st.write('https://www.web1zonasegvrabnonlinedat.club')
#         st.write('https://www.steamcommunnitty.tk')
#         st.write('https://www.doctorannenkova.ru')
#         st.write('https://www.videosfacebook.today')
#         st.write('https://www.dhlexpressmail.com')
#         st.write('https://www.enigreen.net')
#         st.write('https://www.alminarensemble.cl')
#         st.write('https://www.therockacc.org')
#         st.write('https://www.itcitymm.com')
#         st.write('https://www.frysconstucttehe.com')
#         st.write('https://www.skvartremont.ru')
#         st.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE! SO, THE EXAMPLES SHOULD BE UPDATED!')

#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)

#     url = st.text_input('Enter the URL', key='url_input')
#     if st.button("Check URL"):
#         if url:
#             obj = FeatureExtraction(url)
#             x = np.array(obj.getFeaturesList()).reshape(1, 30)

#             # Make predictions using the ensemble model
#             is_phishing = ensemble_model.predict(x)

#             # Display the result
#             if is_phishing:
#                 st.error("This URL is likely unsafe.")
#             else:
#                 st.success("This URL is likely safe.")

#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#4d6cc1'>Mitigating Phishing Risks</h4>", unsafe_allow_html=True)

#     st.write(
#         'Phishing attacks pose significant risks to individuals, businesses, and organizations. They can lead to identity theft, financial loss, data breaches, '
#         'and reputational damage. To protect against phishing attacks, it\'s essential to stay vigilant, be cautious of unsolicited emails or messages, '
#         'verify the authenticity of websites and communications, and regularly update security measures such as antivirus software and firewalls. '
#         'Additionally, education and awareness training for employees and users are crucial in preventing successful phishing attacks.')
#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)

#     # Call the footer function to display the footer
#     footer()


# selected = streamlit_menu()

# if selected == "Home":
#     applicationRun()
# if selected == "Project_Description":
#     Project_Description.show()
#     footer()
# elif selected == "Contact_Us":
#     Contact_Us.show()
#     footer()
# elif selected == "FAQ":
#     FAQ.show()
#     footer()
# # elif selected == "News":
# #     Blog.show()
# #     footer()
# # elif selected == "Blog":
# #     Blog.show()
# #     footer()


# < First Attempt>




import pickle
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split

from feature import FeatureExtraction

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import requests as re
import streamlit as st
from bs4 import BeautifulSoup

# Import the FeatureExtraction class from feature.py
from feature import FeatureExtraction
from menu import streamlit_menu
from pages import FAQ, Contact_Us, Project_Description
from utils import footer

# Set page config
st.set_page_config(page_title='Phishing Website Detection Using Machine Learning', page_icon='./static/favicon.png')
# Add the CSS rule using st.markdown
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
# --- PATH SETTINGS ---
current_dir = Path(_file).parent if "file_" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
phishing_account_pic = current_dir / "static" / "Phishing-account.gif"


import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

#Loading data into dataframe
data = pd.read_csv("phishing.csv")

# droping index column
data = data.drop(['Index'],axis = 1)

# Splitting the dataset into dependent and independent features
X = data.drop(["class"],axis =1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Load the XGBoost model Best Model
with open('pickle/model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Calculate accuracy of XGBoost model (assuming you have test data X_test and labels y_test)
xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))

# Load the XGBoost model
with open('pickle/catboost_model.pkl', 'rb') as f:
    catboost_model = pickle.load(f)

# Calculate accuracy of catboost model (assuming you have test data X_test and labels y_test)
catboost_accuracy = accuracy_score(y_test, catboost_model.predict(X_test))


# Load the decision_tree model
with open('pickle/decision_tree_model.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)

# Calculate accuracy of decision_tree model (assuming you have test data X_test and labels y_test)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_model.predict(X_test))

# Load the knn_model model
with open('pickle/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

# Calculate accuracy of knn_model model (assuming you have test data X_test and labels y_test)
knn_model_accuracy = accuracy_score(y_test, knn_model.predict(X_test))


# Load the logistic_regression model
with open('pickle/logistic_regression_model.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)

# Calculate accuracy of logistic_regression model (assuming you have test data X_test and labels y_test)
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_model.predict(X_test))


# Load the mlp model
with open('pickle/mlp_model.pkl', 'rb') as f:
    mlp_model = pickle.load(f)

# Calculate accuracy of mlp model (assuming you have test data X_test and labels y_test)
mlp_accuracy = accuracy_score(y_test, mlp_model.predict(X_test))


# Load the naive_bayes model
with open('pickle/naive_bayes_model.pkl', 'rb') as f:
    naive_bayes_model = pickle.load(f)

# Calculate accuracy of naive_bayes model (assuming you have test data X_test and labels y_test)
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_model.predict(X_test))


# Load the random_forest model
with open('pickle/random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

# Calculate accuracy of random_forest model (assuming you have test data X_test and labels y_test)
random_forest_accuracy = accuracy_score(y_test, random_forest_model.predict(X_test))


# Load the gradient_boosting model
with open('pickle/gradient_boosting_model.pkl', 'rb') as f:
    gradient_boosting_model = pickle.load(f)

# Calculate accuracy of gradient_boosting model (assuming you have test data X_test and labels y_test)
gradient_boosting_accuracy = accuracy_score(y_test, gradient_boosting_model.predict(X_test))


# Load the svc model
with open('pickle/svc_model.pkl', 'rb') as f:
    svc_model = pickle.load(f)

# Calculate accuracy of svc model (assuming you have test data X_test and labels y_test)
svc_accuracy = accuracy_score(y_test, svc_model.predict(X_test))


def typeWriter():
    typeWriter = """
     <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bootstrap Carousel</title>
        <style>
         /* The typing effect */
         @keyframes typing {
             from { width: 0 }
             to { width: 100% }
         }
         /* The typewriter cursor effect */
         @keyframes blink-caret {
             from, to { border-color: transparent }
             50% { border-color: #4d6cc1 }
         }
         /* The container */
         .typewriter-container {
             overflow: hidden; /* Ensures the text is not revealed until the animation */
             border-right: .15em solid #4d6cc1; /* The typewriter cursor */
             white-space: nowrap; /* Keeps the content on a single line */
             margin: 0 auto; /* Gives that scrolling effect as the typing happens */
             letter-spacing: .15em; /* Adjust as needed */
             animation:
                 typing 3.5s steps(40, end),
                 blink-caret .75s step-end infinite;
         }
     </style>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css">
    </head>
    <body>
        <div class="typewriter-container">
             <h3 style='color:#4d6cc1'>Phish the Phisher before they phish you!!!</h3>
         </div>
    </body>
    </html>
    """
    st.markdown(typeWriter, unsafe_allow_html=True)


def render_bootstrap_carousel():
    # Define the HTML code for the carousel
    html_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bootstrap Carousel</title>
        <style>
         .carousel-container {
             max-width: 850px;
             margin: 0 auto;
             height: 450px; /* Set a fixed height for the carousel */
             overflow: hidden; /* Hide overflow content */
         }
         .carousel-inner img {
             max-height: 100%; /* Ensure images don't exceed the height of the container */
             width: auto; /* Maintain aspect ratio */
             margin: 0 auto; /* Center images horizontally */
         }
     </style>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css">
    </head>
    <body>
        <div class="container">
            <!-- Carousel -->
            <div id="myCarousel" class="carousel slide" data-ride="carousel">
                <!-- Indicators -->
                <ol class="carousel-indicators">
                    <li data-target="#myCarousel" data-slide-to="0" class="active"></li>
                    <li data-target="#myCarousel" data-slide-to="1"></li>
                    <li data-target="#myCarousel" data-slide-to="2"></li>
                </ol>

                <!-- Slides -->
                <div class="carousel-inner">
                    <div class="carousel-item active">
                        <img class="d-block w-100" src="https://cdn.pixabay.com/photo/2023/05/26/00/23/hacker-8018467_1280.png" alt="First slide">
                    </div>
                    <div class="carousel-item">
                        <img class="d-block w-100" src="https://cdn.pixabay.com/photo/2023/09/18/07/07/ai-generated-8259795_1280.jpg" alt="Second slide">
                    </div>
                    <div class="carousel-item">
                        <img class="d-block w-100" src="https://cdn.pixabay.com/photo/2023/05/26/00/55/hacker-8018499_1280.png" alt="Third slide">
                    </div>
                </div>

                <!-- Controls -->
                <a class="carousel-control-prev" href="#myCarousel" role="button" data-slide="prev">
                    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                    <span class="sr-only">Previous</span>
                </a>
                <a class="carousel-control-next" href="#myCarousel" role="button" data-slide="next">
                    <span class="carousel-control-next-icon" aria-hidden="true"></span>
                    <span class="sr-only">Next</span>
                </a>
            </div>
        </div>

        <!-- JavaScript -->
        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js"></script>

        <script>
            // Initialize the carousel with an interval of 2 seconds
            $('#myCarousel').carousel({
                interval: 2000
            });

            // Hook into the slide event
            $('#myCarousel').on('slide.bs.carousel', function (e) {
                // Do something when the carousel slide event occurs
                console.log('Slide event occurred:', e);
            });

            // Hook into the slid event
            $('#myCarousel').on('slid.bs.carousel', function (e) {
                // Do something when the carousel slid event occurs
                console.log('Slid event occurred:', e);
            });
        </script>
    </body>
    </html>
    """
    # Render the HTML code in Streamlit
    st.components.v1.html(html_code, width=600, height=330)
    

class EnsembleModel(BaseEstimator, ClassifierMixin):
    def _init_(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        # Combine predictions (e.g., by averaging or voting)
        # For simplicity, let's use majority voting
        return np.mean(predictions, axis=0) > 0.5  # Use threshold as needed


# Load the trained model
file = open("pickle/ensemble_model.pkl", "rb")
model = pickle.load(file)
file.close()


# Calculate accuracy of ensemble model (assuming you have test data X_test and labels y_test)
ensemble_accuracy = accuracy_score(y_test, model.predict(X_test))
def applicationRun():

    st.markdown("<h1 style='color:#c8a808'>Phishr</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    typeWriter()

    # Call the function to render the carousel
    render_bootstrap_carousel()
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4d6cc1'>Understanding Phishing Attack</h4>", unsafe_allow_html=True)
    st.write('Phishing attacks are a common type of cyber attack where malicious actors attempt to deceive individuals or organizations into revealing '
             'sensitive information such as usernames, passwords, credit card numbers, or other personal or financial data. These attacks typically '
             'involve impersonating a trusted entity, such as a bank, a government agency, a company, or even a colleague or friend.')

    # Load the GIF
    phishing_acc = "static/Phishing-account.gif"

    # Display the GIF
    st.image(phishing_acc, caption='PHISHr', use_column_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander('EXAMPLE LEGITIMATE URLs:'):
        st.write('https://www.twitter.com')
        st.write('https://www.youtube.com')
        st.write('https://www.google.com')
        st.write('https://www.facebook.com')
        st.write('https://www.instagram.com')
        st.write('https://www.icicibank.com')
        st.write('https://graphicriver.net')
        st.write('https://www.sourceforge.net')
        st.write('https://www.metro.co.uk')
        st.write('https://www.stackoverflow.com')
        st.write('https://www.olx.co.id')
        st.write('https://www.indianexpress.com')
        st.write('https://www.kenh14.vn')
        st.write('https://www.otomoto.pl')
        st.write('https://www.askubuntu.com')
        st.caption('This links can be used !!!')

    with st.expander('EXAMPLE PHISHING URLs:'):
        st.write('https://www.kinemind.com')
        st.write('https://www.eevee.tv')
        st.write('https://www.grandcup.xyz')
        st.write('https://www.hamt.jp')
        st.write('https://www.web1zonasegvrabnonlinedat.club')
        st.write('https://www.steamcommunnitty.tk')
        st.write('https://www.doctorannenkova.ru')
        st.write('https://www.videosfacebook.today')
        st.write('https://www.dhlexpressmail.com')
        st.write('https://www.enigreen.net')
        st.write('https://www.alminarensemble.cl')
        st.write('https://www.therockacc.org')
        st.write('https://www.itcitymm.com')
        st.write('https://www.frysconstucttehe.com')
        st.write('https://www.skvartremont.ru')
        st.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE! SO, THE EXAMPLES SHOULD BE UPDATED!')

    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)

    url = st.text_input('Enter the URL', key='url_input')
    # Create a dropdown menu to select the model
    model_selection = st.selectbox('Select Model', ['XGBoost','Catboost','Decision Tree','Gradient Boosting','Knn','Logistic Regression','MLP','Naive Bayes','Random Forest','SVC','Ensemble'])

    if st.button("Check URL"):
        if url:
            obj = FeatureExtraction(url)
            x = np.array(obj.getFeaturesList()).reshape(1, 30)
            # Make predictions based on selected model
            if model_selection == 'XGBoost':
                is_phishing = model.predict(x)
                if is_phishing:
                    st.error("It is unsafe to go")
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.image("static/Warning.gif", caption='Warning', use_column_width=True)
                    st.write(f'Accuracy of XGBoost model: {xgb_accuracy}')
                    st.snow()
                else:
                    st.success("It is safe to go")
                    st.image("static/Safe.gif", caption='Safe', use_column_width=True)
                    st.write(f'Accuracy of XGBoost model: {xgb_accuracy}')
                    st.balloons()
            elif model_selection == 'Ensemble':
                is_phishing = model.predict(x)
                if is_phishing:
                    st.error("This URL is likely unsafe.")
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.image("static/Warning.gif", caption='Warning', use_column_width=True)
                    st.write(f'Accuracy of Ensemble model: 0.9516056083220262')
                    st.snow()
                else:
                    st.success("This URL is likely safe.")
                    st.image("static/Safe.gif", caption='Safe', use_column_width=True)
                    st.write(f'Accuracy of Ensemble model: 0.9516056083220262')
                    st.balloons()
            elif model_selection == 'Catboost':
                is_phishing = model.predict(x)
                if is_phishing:
                    st.error("This URL is likely unsafe.")
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.image("static/Warning.gif", caption='Warning', use_column_width=True)
                    st.write(f'Accuracy of Catboost model: {catboost_accuracy}')
                    st.snow()
                else:
                    st.success("This URL is likely safe.")
                    st.image("static/Safe.gif", caption='Safe', use_column_width=True)
                    st.write(f'Accuracy of Catboost model: {catboost_accuracy}')
                    st.balloons()
            elif model_selection == 'Decision Tree':
                is_phishing = model.predict(x)
                if is_phishing:
                    st.error("This URL is likely unsafe.")
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.image("static/Warning.gif", caption='Warning', use_column_width=True)
                    st.write(f'Accuracy of Decision tree model: {decision_tree_accuracy}')
                    st.snow()
                else:
                    st.success("This URL is likely safe.")
                    st.image("static/Safe.gif", caption='Safe', use_column_width=True)
                    st.write(f'Accuracy of Decision tree model: {decision_tree_accuracy}')
                    st.balloons()
            elif model_selection == 'Gradient Boosting':
                is_phishing = model.predict(x)
                if is_phishing:
                    st.error("This URL is likely unsafe.")
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.image("static/Warning.gif", caption='Warning', use_column_width=True)
                    st.write(f'Accuracy of Gradient Boosting model: {gradient_boosting_accuracy}')
                    st.snow()
                else:
                    st.success("This URL is likely safe.")
                    st.image("static/Safe.gif", caption='Safe', use_column_width=True)
                    st.write(f'Accuracy of Gradient Boosting model: {gradient_boosting_accuracy}')
                    st.balloons()
            elif model_selection == 'Knn':
                is_phishing = model.predict(x)
                if is_phishing:
                    st.error("This URL is likely unsafe.")
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.image("static/Warning.gif", caption='Warning', use_column_width=True)
                    st.write(f'Accuracy of Knn model: {knn_model_accuracy}')
                    st.snow()
                else:
                    st.success("This URL is likely safe.")
                    st.image("static/Safe.gif", caption='Safe', use_column_width=True)
                    st.write(f'Accuracy of Knn model: {knn_model_accuracy}')
                    st.balloons()
            elif model_selection == 'Logistic Regression':
                is_phishing = model.predict(x)
                if is_phishing:
                    st.error("This URL is likely unsafe.")
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.image("static/Warning.gif", caption='Warning', use_column_width=True)
                    st.write(f'Accuracy of Logistic Regression model: {logistic_regression_accuracy}')
                    st.snow()
                else:
                    st.success("This URL is likely safe.")
                    st.image("static/Safe.gif", caption='Safe', use_column_width=True)
                    st.write(f'Accuracy of  Logistic Regression model: {logistic_regression_accuracy}')
                    st.balloons()
            elif model_selection == 'MLP':
                is_phishing = model.predict(x)
                if is_phishing:
                    st.error("This URL is likely unsafe.")
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.image("static/Warning.gif", caption='Warning', use_column_width=True)
                    st.write(f'Accuracy of MLP model: {mlp_accuracy}')
                    st.snow()
                else:
                    st.success("This URL is likely safe.")
                    st.image("static/Safe.gif", caption='Safe', use_column_width=True)
                    st.write(f'Accuracy of MLP model: {mlp_accuracy}')
                    st.balloons()
            elif model_selection == 'Naive Bayes':
                is_phishing = model.predict(x)
                if is_phishing:
                    st.error("This URL is likely unsafe.")
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.image("static/Warning.gif", caption='Warning', use_column_width=True)
                    st.write(f'Accuracy of Naive Bayes model: {naive_bayes_accuracy}')
                    st.snow()
                else:
                    st.success("This URL is likely safe.")
                    st.image("static/Safe.gif", caption='Safe', use_column_width=True)
                    st.write(f'Accuracy of Naive Bayes model: {naive_bayes_accuracy}')
                    st.balloons()
            elif model_selection == 'Random Forest':
                is_phishing = model.predict(x)
                if is_phishing:
                    st.error("This URL is likely unsafe.")
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.image("static/Warning.gif", caption='Warning', use_column_width=True)
                    st.write(f'Accuracy of Random Forest model: {random_forest_accuracy}')
                    st.snow()
                else:
                    st.success("This URL is likely safe.")
                    st.image("static/Safe.gif", caption='Safe', use_column_width=True)
                    st.write(f'Accuracy of Random Forest model: {random_forest_accuracy}')
                    st.balloons()
            elif model_selection == 'SVC':
                is_phishing = model.predict(x)
                if is_phishing:
                    st.error("This URL is likely unsafe.")
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.image("static/Warning.gif", caption='Warning', use_column_width=True)
                    st.write(f'Accuracy of SVC model: {svc_accuracy}')
                    st.snow()
                else:
                    st.success("This URL is likely safe.")
                    st.image("static/Safe.gif", caption='Safe', use_column_width=True)
                    st.write(f'Accuracy of SVC model: {svc_accuracy}')
                    st.balloons()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4d6cc1'>Mitigating Phishing Risks</h4>", unsafe_allow_html=True)

    st.write(
        'Phishing attacks pose significant risks to individuals, businesses, and organizations. They can lead to identity theft, financial loss, data breaches, '
        'and reputational damage. To protect against phishing attacks, it\'s essential to stay vigilant, be cautious of unsolicited emails or messages, '
        'verify the authenticity of websites and communications, and regularly update security measures such as antivirus software and firewalls. '
        'Additionally, education and awareness training for employees and users are crucial in preventing successful phishing attacks.')
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)

    # Call the footer function to display the footer
    footer()


selected = streamlit_menu()

if selected == "Home":
    applicationRun()
if selected == "Project_Description":
    Project_Description.show()
    footer()
elif selected == "Contact_Us":
    Contact_Us.show()
    footer()
elif selected == "FAQ":
    FAQ.show()
    footer()
