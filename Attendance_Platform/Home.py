import streamlit as st


st.set_page_config(page_title='Attendance System',layout='wide')

st.header('Attendance System Using Face Recognition')

with st.spinner("Loading Models and Connecting to Redis db ..."):
  import face_rec
  
st.success("Model loaded sucessfully")
st.success("Redis db sucessfully connected")

# Main content
st.subheader("Welcome to the Attendance System!")
st.write("This system is designed to simplify attendance tracking in educational institutions and organizations using cutting-edge face recognition technology.")

# Get Started
st.header('Get Started:')
st.write("Ready to streamline attendance tracking in your organization? Start by navigating to the 'Add Registration' page to register users.")

# Features
st.header('Key Features:')
st.write("1. **Real-time Face Prediction:** Capture attendance by recognizing faces in real-time during classes, meetings, or events.")
st.write("2. **Efficient Registration:** Easily register students and teachers with their name and role. You can also include additional information like student ID or department.")
st.write("3. **Reporting Log:** Keep a detailed log of attendance records, making it easy to track attendance trends and analyze data over time.")
st.write("4. **User Management:** Delete registration data when needed to manage user information efficiently.")
st.write("5. **Customizable Settings:** Tailor the system to your theme, either light or dark mode.")

# How to Use
st.header('How to Use:')
st.write("1. **Add Registration:** Begin by registering students and teachers. Visit the 'Add Registration' page to add users to the system.")
st.write("2. **Real-time Face Prediction:** Navigate to the 'Real-time Face Prediction' page to take attendance in real-time.")
st.write("3. **Reporting Log:** Access the 'Reporting Log' page to view attendance records and generate reports.")
st.write("4. **Delete Registration:** If needed, use the 'Delete Registration' page to manage user data.")
st.write("5. **Settings:** Customize the system on the 'Settings' page according to your desired dark or light mode theme.")

# Security and Privacy
st.header('Security and Privacy:')
st.write("I do prioritize security and privacy. And so the facial data used for attendance tracking is securely stored and not shared with third parties.")
st.write("Additionally, our system adheres to all relevant data protection regulations.")

# Support and Contact
st.header('Support and Contact:')
st.write("If you have any questions or need assistance, I am here to help. Contact me at robertwinstonwidjaja1@gmail.com for prompt assistance.")

