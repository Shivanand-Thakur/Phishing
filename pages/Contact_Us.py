
# import matplotlib.pyplot as plt
# import requests as re
# import streamlit as st
# from bs4 import BeautifulSoup

# from utils import footer

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

#     st.markdown("<h1 style='color:#c8a808'>Contact Us</h1>", unsafe_allow_html=True)
#     st.write("Have questions, feedback, or collaboration opportunities? "
#              "Feel free to reach out to us using the contact form below "
#              "or through our social media channels.")
#     # Add content for the About Us page here
#     st.header(":mailbox: Get In Touch With Me!")
#     contact_form = """
#     <form action="https://formsubmit.co/adarshvajpayee19@gmail.com" method="POST">
#          <input type="hidden" name="_captcha" value="false">
#          <input type="text" name="name" placeholder="Your name" required>
#          <input type="email" name="email" placeholder="Your email" required>
#          <textarea name="message" placeholder="Your message here"></textarea>
#          <button type="submit">Send</button>
#     </form>
#     """

#     st.markdown(contact_form, unsafe_allow_html=True)
#     # Use Local CSS File
#     def local_css(file_name):
#         with open(file_name) as f:
#             st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


#     local_css("styles/style.css")

#     st.markdown("<h3 style='color:#4d6cc1'>Connect with Us on Social Media</h3>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#c8a808'>Follow us on social media for updates, news, and more!</h4>", unsafe_allow_html=True)

#     # Call the footer function to display the footer
#     # footer()


import streamlit as st

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
    st.markdown("<h1 style='color:#c8a808'>Contact Us</h1>", unsafe_allow_html=True)
    st.write("Have questions, feedback, or collaboration opportunities? "
             "Feel free to reach out to us using the contact form below "
             "or through our social media channels.")
    
    # Add content for the About Us page here
    st.header(":mailbox: Get In Touch With Me!")
    
    # Define the HTML form with email validation script
    contact_form = """
    <form id="contactForm" action="https://formsubmit.co/adarshvajpayee19@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" id="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit" id="submitButton">Send</button>
    </form>
    <script>
        document.getElementById('contactForm').onsubmit = function() {
            var email = document.getElementById('email').value;
            if (email.indexOf('@') == -1) {
                alert('Please enter a valid email address.');
                return false; // Prevent form submission
            } else {
                alert('Email sent successfully!');
                return true; // Allow form submission
            }
        }
    </script>
    """
    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    local_css("styles/style.css")

    # Display the HTML form
    st.markdown(contact_form, unsafe_allow_html=True)

