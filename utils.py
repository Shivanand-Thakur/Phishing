import streamlit as st


# Define the footer layout
def footer():
    st.markdown("""
    <style>
    footer{
        background-color: #e4dd86;
    }
    .footerContainer{
        width: 100%;
        padding: 70px 30px 20px ;
    }
    .socialIcons{
        display: flex;
        justify-content: center;
    }
    .socialIcons a{
        text-decoration: none;
        padding:  10px;
        background-color: white;
        margin: 10px;
        border-radius: 50%;
    }
    .socialIcons a i{
        font-size: 2em;
        color: black;
        opacity: 0.9;
    }
    /* Hover affect on social media icon */
    .socialIcons a:hover{
        background-color: #111;
        transition: 0.5s;
    }
    .socialIcons a:hover i{
        color: white;
        transition: 0.5s;
    }
    .footerNav{
        margin: 30px 0;
    }
    .footerNav ul{
        display: flex;
        justify-content: center;
        list-style-type: none;
    }
    .footerNav ul li a{
        color:black;
        margin: 3px;
        text-decoration: none;
        font-size: 1.3em;
        opacity: 0.7;
        transition: 0.5s;
    }
    .footerNav ul li a:hover{
        opacity: 1;
    }
    .footerBottom{
        background-color: #000;
        padding: 20px;
        text-align: center;
    }
    .footerBottom p{
        color: white;
    }
    .designer{
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 400;
        margin: 0px 5px;
    }
    @media (max-width: 700px){
        .footerNav ul{
            flex-direction: column;
        }
        .footerNav ul li{
            width:100%;
            text-align: center;
            margin: 6px;
        }
        .socialIcons a{
            padding: 8px;
            margin: 4px;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
</head>
<body>
    <footer>
    <div class="footerContainer">
        <div class="socialIcons">
            <a href=""><i class="fa-brands fa-facebook"></i></a>
            <a href=""><i class="fa-brands fa-instagram"></i></a>
            <a href=""><i class="fa-brands fa-twitter"></i></a>
            <a href=""><i class="fa-brands fa-google-plus"></i></a>
            <a href=""><i class="fa-brands fa-youtube"></i></a>
        </div>
        <div class="footerNav">
            <ul><li><a href="app.py">Home</a></li>
                <li><a href="Blog.py">News</a></li>
                <li><a href="">About</a></li>
                <li><a href="">Contact Us</a></li>
                <li><a href="">our Team</a></li>
            </ul>
        </div>
        <div class="footerBottom">
            <p>Copyright &copy;2024; Let's Catch The <span class="designer">Phishr</span></p>
        </div>
    </div>
    </footer>
</body>
</html>
    """, unsafe_allow_html=True)
