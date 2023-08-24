import streamlit as st


st.set_page_config(page_title='Attendance System',layout='wide')

st.header('Attendance System Using Face Recognition')

with st.spinner("Loading Models and Connecting to Redis db ..."):
  import face_rec
  
st.success("Model loaded sucessfully")
st.success("Redis db sucessfully connected")