# Load all environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import streamlit as st  # Import Streamlit for web app development
import os               # Import os for interacting with the operating system
from PIL import Image   # Import Image from PIL to handle image files
import google.generativeai as genai  # Import Google's generative AI module

# Configure the Google generative AI with an API key obtained from the environment variables
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize a generative model from Google's generative AI library
model = genai.GenerativeModel('gemini-pro-vision')

# Function to get response from the Gemini generative model
def get_gemini_response(input, image, prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text

# Function to process the uploaded image file
def image_input_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()  # Get the binary content of the uploaded file

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the MIME type of the uploaded file
                "data": bytes_data                # Include the binary data of the image
            }
        ]

        return image_parts

    else:
        raise FileNotFoundError('No file uploaded')  # Raise an error if no file is uploaded

# Set up the Streamlit page configuration
st.set_page_config(page_title='Multi-Language Invoice Extractor')
st.header('Multi-Language Invoice Extractor')  # Display a header on the page

# Create an input field for text prompts
input = st.text_input('Enter your prompt ...')

# Create a file uploader widget for image files
uploaded_file = st.file_uploader('Choose an image of the invoice', type=['jpg', 'jpeg', 'png'])

image = ''
if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Open the uploaded image file
    st.image(image, caption='Uploaded Image', use_column_width=True)  # Display the uploaded image

# Create a button for submitting the information
submit = st.button('Tell me about the image')

# Predefined prompt for the generative model
input_prompt = """
                You are an expert in understanding invoices.
                You will receive input images as invoices & you will have answer questions based on input image.
               """

# Handling the submit action
if submit:
    image_data = image_input_setup(uploaded_file)  # Process the uploaded image
    response = get_gemini_response(input_prompt, image_data, input)  # Get the model's response
    st.subheader('The response is ....')  # Display a subheader
    st.success(response)  # Show the model's response
