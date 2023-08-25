import streamlit as st 
from Home import face_rec
import cv2 
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
import os

from twilio.rest import Client

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = os.environ['AC022c4944e9902bf04423ae08f4703c3e']
auth_token = os.environ['a44bcaf3cf5c9aa64566730ffd6e2656']
client = Client(account_sid, auth_token)

token = client.tokens.create()


st.set_page_config(page_title="Registration Form",layout='centered')
st.subheader("Registration Form")

## Instantiate Registration Form
registration_form = face_rec.RegistrationForm()

# Collect Person name and role
# Form
person_name = st.text_input(label='Name',placeholder='First & Last Name')
role = st.selectbox(label='Select your Role', options=('Student','Teacher'))


# Collect Facial embeddings of that person
def video_callback_func(frame):
  img = frame.to_ndarray(format='bgr24') # 3d array bgr
  reg_img, embedding = registration_form.get_embedding(img)
  
  # Save data into local computer txt
  if embedding is not None:
    with open('face_embedding.txt', mode='ab') as f:
      np.savetxt(f,embedding)
  
  return av.VideoFrame.from_ndarray(reg_img,format='bgr24')

webrtc_streamer(
  rtc_configuration={
      "iceServers": token.ice_servers
  },
  key='registration',
  video_frame_callback=video_callback_func)

# Save the data in redis database

if st.button('Submit'):
  return_val = registration_form.save_data_in_redis_db(person_name,role)
  if return_val == True:
    st.success(f"{person_name} registered sucessfully")
  elif return_val == 'name_false':
    st.error('Please enter the name: Name cannot be empty or spaces')
  
  elif return_val == 'file_false':
    st.error('face_embedding.txt is not found. Please refresh the page and execute again.')
    