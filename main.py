import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to make predictions
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_plant_disease_model.keras')
    
    # Convert the uploaded file (test_image) to an Image object
    image = Image.open(test_image)
    
    # Resize image and preprocess it for the model
    image = image.resize((128, 128))
    input_arr = np.array(image)  # Convert to numpy array
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch
    
    # Make predictions using the model
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)  # Find the index of the class with the highest probability
    return result_index

# Custom CSS Styling
st.markdown("""
    <style>
        /* General background for the app */
        .stApp {
            background-color: #F9F9F9;
        }

        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #F4F4F9;
        }

        /* Header styling */
        h1, h2 {
            color: #2E8B57;
            font-family: 'Helvetica', sans-serif;
        }

        /* Styling for the 'Show Image' and 'Predict' buttons */
        .stButton > button {
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
        }

        /* Green button for 'Show Image' */
        .stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
        }

        /* Blue button for 'Predict' */
        .stButton > button:last-child {
            background-color: #2196F3;
            color: white;
        }

        /* Section background color */
        .section-home {
            background-color: #e6f7f3;
            padding: 20px;
            border-radius: 8px;
        }

        .section-about {
            background-color: #fff9e6;
            padding: 20px;
            border-radius: 8px;
        }

        .section-disease-recognition {
            background-color: #f9e6f7;
            padding: 20px;
            border-radius: 8px;
        }

        /* Styling for success message */
        .stSuccess {
            color: #28a745;
            font-weight: bold;
        }

        /* Section Text Styling */
        .section-text {
            font-size: 16px;
            font-family: 'Arial', sans-serif;
            color: #444;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page Section
if app_mode == "Home":
    st.markdown('<div class="section-home">', unsafe_allow_html=True)
    st.header("PLANT DISEASE CLASSIFICATION SYSTEM")
    image_path = "home_page.png"  # You may need to fix this path if it's incorrect
    st.image(image_path, use_container_width=True)
    st.markdown("""
    <div class="section-text">
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# About Page Section
elif app_mode == "About":
    st.markdown('<div class="section-about">', unsafe_allow_html=True)
    st.header("About")
    st.markdown("""
    <div class="section-text">
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.
    A new directory containing 33 test images was created later for prediction purposes.

    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Disease Recognition Page Section
elif app_mode == "Disease Recognition":
    st.markdown('<div class="section-disease-recognition">', unsafe_allow_html=True)
    st.header("Disease Recognition")
    
    # File uploader for image upload
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        # Show "Show Image" button
        if st.button("Show Image"):
            st.image(test_image, use_container_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            # Reading labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']
            
            # Display the result
            st.success(f"Model Prediction: It's a {class_name[result_index]}")
    st.markdown('</div>', unsafe_allow_html=True) 