import hashlib

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
import streamlit as st
from streamlit_option_menu import option_menu
# import nlp as nlp
from wordcloud import WordCloud

# import spacy
from utils import footer


# Function to hash the password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname="phishing_database",
    user="postgres",
    password="adarshvajpayee@192001",
    host="localhost"
)
conn.autocommit = True  # Enable autocommit mode
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


matplotlib.use('Agg')
# Functions
# Avatar Image using a url
avatar1 = "https://www.w3schools.com/howto/img_avatar1.png"
avatar2 = "https://www.w3schools.com/howto/img_avatar2.png"

def readingTime(mytext):
    total_words = len([token for token in mytext.split(" ")])
    estimatedTime = total_words / 200.0
    return estimatedTime


# Layout Templates
title_temp = """
	<div style="background-color:#b5b5b5;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">{}</h1>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<h6>Author:{}</h6>
	<br/>
	<br/>
	<p style="text-align:justify">{}</p>
	</div>
	"""
article_temp = """
	<div style="background-color:#b5b5b5;padding:10px;border-radius:5px;margin:10px;">
	<h4 style="color:white;text-align:center;">{}</h1>
	<h6 style="color:white;text-align:center;">Author:{}</h6> 
	<h6 style="color:white;text-align:center;">Post Date: {}</h6>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>
	<p style="text-align:justify">{}</p>
	</div>
	"""
head_message_temp = """
	<div style="background-color:#464e5f;padding:10px;border-radius:5px;margin:10px;">
	<h4 style="color:white;text-align:center;">{}</h1>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;">
	<h6 style="color:white;text-align:center;">Author:{}</h6> 		
	<h6 style="color:white;text-align:center;">Post Date: {}</h6>		
	</div>
	"""
full_message_temp = """
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<p style="text-align:justify;color:black;padding:10px">{}</p>
	</div>
	"""

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""


# Functions for database operations
def create_table():
    with conn.cursor() as cursor:
        cursor.execute('CREATE TABLE IF NOT EXISTS blogtable(author TEXT,title TEXT,article TEXT,postdate DATE)')

def add_data(author, title, article, postdate):
    with conn.cursor() as cursor:
        cursor.execute('INSERT INTO blogtable(author,title,article,postdate) VALUES (%s, %s, %s, %s)',
                       (author, title, article, postdate))

def view_all_notes():
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM blogtable')
        data = cursor.fetchall()
    return data

def view_all_titles():
    with conn.cursor() as cursor:
        cursor.execute('SELECT DISTINCT title FROM blogtable')
        data = cursor.fetchall()
    return data

def get_blog_by_title(title):
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT * FROM blogtable WHERE title=%s', (title,))
            data = cursor.fetchall()
        return data
    except psycopg2.Error as e:
        print("Error fetching blog by title:", e)
        return None



def get_blog_by_author(author):
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM blogtable WHERE author=%s', (author,))
        data = cursor.fetchall()
    return data

def delete_data(title):
    with conn.cursor() as cursor:
        cursor.execute('DELETE FROM blogtable WHERE title=%s', (title,))

def create_usertable():
    with conn.cursor() as cursor:
        cursor.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username, password):
    with conn.cursor() as cursor:
        cursor.execute('INSERT INTO userstable(username,password) VALUES (%s,%s)', (username, password))

def login_user(username, password):
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM userstable WHERE username=%s AND password=%s', (username, password))
        data = cursor.fetchall()
    return data

def create_admin_table():
    with conn.cursor() as cursor:
        cursor.execute('CREATE TABLE IF NOT EXISTS admin_table (admin_username TEXT, admin_password TEXT)')

def insert_admin_credentials():
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM admin_table')
        data = cursor.fetchall()
        if not data:  # Check if there are no admin credentials in the database
            hashed_password = hash_password('adarsh@123')  # Hash the password
            cursor.execute("INSERT INTO admin_table (admin_username, admin_password) VALUES (%s, %s)", ('adarsh vajpayee', hashed_password))
            hashed_password = hash_password('prajwal@123')  # Hash the password
            cursor.execute("INSERT INTO admin_table (admin_username, admin_password) VALUES (%s, %s)", ('prajwal c', hashed_password))
            print("Admin credentials inserted successfully.")
        else:
            print("Admin credentials already exist.")

def login_admin(username, password):
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM admin_table WHERE admin_username=%s', (username,))
        data = cursor.fetchone()
        if data:
            hashed_password = data[1]
            if hashed_password == hash_password(password):
                print("Admin login successful.")
                return True
        print("Admin login failed. Invalid username or password.")
        return False

# Call the function to create the admin table and insert admin credentials
create_admin_table()
insert_admin_credentials()

create_usertable()

def view_all_users():
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM userstable')
        data = cursor.fetchall()
    conn.close()
    return data

# Functions for managing the blog
def manage_blog():
    admin_username = st.sidebar.text_input("Admin Username")
    admin_password = st.sidebar.text_input("Admin Password", type="password")
    if st.sidebar.checkbox("Admin Login"):
        # # Check admin credentials using login_user function
        # conn = sqlite3.connect('data.db')
        # hashed_password = hash_password(admin_password)  # Hash the input password
        # admin_data = login_admin(admin_username, hashed_password)  # Compare with hashed password in the database
        # conn.close()
        # admin_data = login_admin(admin_username, hash_password(admin_password))
        if login_admin(admin_username, admin_password):
            st.markdown("<h3 style='color:#4d6cc1'>Manage Blog</h3>", unsafe_allow_html=True)
            st.success("Logged in as Admin")
            result = view_all_notes()
            clean_db = pd.DataFrame(result, columns=["Author", "Title", "Article", "Date"])
            st.dataframe(clean_db)
            unique_list = [i[0] for i in view_all_titles()]
            delete_by_title = st.selectbox("Select Title", unique_list)
            if st.button("Delete"):
                delete_data(delete_by_title)
                st.warning("Deleted: '{}'".format(delete_by_title))

            if st.checkbox("Metrics"):
                new_df = clean_db
                new_df['Length'] = new_df['Article'].str.len()
                st.dataframe(new_df)
                st.markdown("<h3 style='color:#4d6cc1'>Author Stats</h3>", unsafe_allow_html=True)
                st.bar_chart(new_df['Author'].value_counts())
                st.markdown("<h3 style='color:#4d6cc1'>Author Stats (Pie Chart)</h3>", unsafe_allow_html=True)
                st.write(new_df['Author'].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()

            # if st.checkbox("WordCloud"):
            #     st.markdown("<h3 style='color:#4d6cc1'>Word Cloud</h3>", unsafe_allow_html=True)
            #     text = ', '.join(clean_db['Article'])
            #     wordcloud = WordCloud().generate(text)
            #     plt.imshow(wordcloud, interpolation='bilinear')
            #     plt.axis("off")
            #     st.pyplot()

            if st.checkbox("BarH Plot"):
                st.markdown("<h3 style='color:#4d6cc1'>Length of Articles</h3>", unsafe_allow_html=True)
                new_df = clean_db
                new_df['Length'] = new_df['Article'].str.len()
                barh_plot = new_df.plot.barh(x='Author', y='Length', figsize=(10, 10))
                st.write(barh_plot)
                st.pyplot()
        else:
            st.warning("Incorrect Admin Username or Password")
    else:
        st.warning("Please login as Admin")
    # Call the footer function to display the footer
    # footer()


def show():
    # UI
    # st.title("Securing Login Apps Against SQL Injection")
    # st.title("<span style='color:#4d6cc1'>Securing Login Apps Against SQL Injection</span>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:#c8a808'>Phishing Blog</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    safe = """
    <span style="color:#4d6cc1"><b>Stay Safe Online:</b></span>  Watch out for fake emails and websites trying to trick you into sharing personal info. Check web addresses, be cautious with emails, and avoid clicking unknown links. Vigilance is key to staying safe from phishing scams.
    """
    description = """
    <span style="color:#4d6cc1"><b>Unveiling the Phishing Threat:</b></span> Dive into the world of phishing attacks and learn how cybercriminals manipulate unsuspecting users. Explore common tactics used to deceive individuals into revealing sensitive information. Discover essential tips to identify and avoid falling victim to phishing scams. Stay informed and arm yourself with the knowledge to protect your digital identity.
    """

    st.markdown(safe, unsafe_allow_html=True)
    st.markdown(description, unsafe_allow_html=True)

    # Display menu options based on user role
    menu = ["Login", "SignUp", "Admin Login"]
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == "Login":
        create_table()  # Ensure table exists before attempting to fetch data
        st.markdown("<hr>",unsafe_allow_html=True)
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            hashed_password = hash_password(password)  # Hash the input password
            result = login_user(username, hashed_password)  # Compare with hashed password in the database
            if result:
                st.success("Logged In as {}".format(username))
                html_temp = """
                <div style="background-color:{};padding:10px;border-radius:10px">
                <h1 style="color:{};text-align:center;">Phishing Blog </h1>
                </div>
                """
                st.markdown(html_temp.format('#a3a7cf', 'white'), unsafe_allow_html=True)
                menu = ["Home", "View Post", "Add Post", "Search"]
                choice = st.sidebar.selectbox("Menu", menu)

                if choice == "Home":
                    st.markdown("<h3 style='color:#4d6cc1'>Blogs</h3>", unsafe_allow_html=True)

                    result = view_all_notes()
                    for i in result:
                        short_article = str(i[2])[0:50]
                        st.write(title_temp.format(i[1], i[0], short_article), unsafe_allow_html=True)

                elif choice == "View Post":
                    st.markdown("<h3 style='color:#4d6cc1'>View Post</h3>", unsafe_allow_html=True)

                    all_titles = [i[0] for i in view_all_titles()]
                    postlist = st.sidebar.selectbox("Posts", all_titles)
                    post_result = get_blog_by_title(postlist)
                    for i in post_result:
                        st.text("Reading Time:{} minutes".format(readingTime(str(i[2]))))
                        st.markdown(head_message_temp.format(i[1], i[0], i[3]), unsafe_allow_html=True)
                        st.markdown(full_message_temp.format(i[2]), unsafe_allow_html=True)

                elif choice == "Add Post":
                    st.markdown("<h3 style='color:#4d6cc1'>Add Your Article</h3>", unsafe_allow_html=True)
                    create_table()
                    blog_title = st.text_input('Enter Post Title')
                    blog_author = st.text_input("Enter Author Name", max_chars=50)
                    blog_article = st.text_area("Enter Your Message", height=200)
                    blog_post_date = st.date_input("Post Date")
                    if st.button("Add"):
                        add_data(blog_author, blog_title, blog_article, blog_post_date)
                        st.success("Post::'{}' Saved".format(blog_title))



                elif choice == "Search":
                    st.markdown("<h3 style='color:#4d6cc1'>Search Articles</h3>", unsafe_allow_html=True)
                    search_term = st.text_input("Enter Term")
                    search_choice = st.radio("Field to Search", ("title", "author"))
                    if st.button('Search'):
                        if search_choice == "title":
                            article_result = get_blog_by_title(search_term)
                        elif search_choice == "author":
                            article_result = get_blog_by_author(search_term)

                        # Preview Articles
                        for i in article_result:
                            st.text("Reading Time:{} minutes".format(readingTime(str(i[2]))))
                            # st.write(article_temp.format(i[1],i[0],i[3],i[2]),unsafe_allow_html=True)
                            st.write(head_message_temp.format(i[1], i[0], i[3]), unsafe_allow_html=True)
                            st.write(full_message_temp.format(i[2]), unsafe_allow_html=True)
            else:
                st.warning("Incorrect Username/Password")
        # Call the footer function to display the footer
        # footer()

    elif choice == "SignUp":
        st.markdown("<h3 style='color:#4d6cc1'>Create An Account</h3>", unsafe_allow_html=True)
        new_username = st.text_input("User name")
        new_password = st.text_input("Password", type='password', key='password_input')
        confirm_password = st.text_input('Confirm Password', type='password')

        if new_password != '' and confirm_password != '':
            if new_password == confirm_password:
                hashed_password = hash_password(new_password)  # Hash the password
                # add_userdata(new_username, hashed_password)  # Store hashed password in the database
                st.success("Valid Password Confirmed")
                if st.button("Sign Up"):
                    add_userdata(new_username, hashed_password)
                    # The sign-up button should perform no further action
                    st.success("Successfully Created an Account")
            else:
                st.warning("Password not the same")
        else:
            st.warning("Please fill in both password fields")
        # Call the footer function to display the footer
        # footer()


    elif choice == "Admin Login":
        manage_blog()


