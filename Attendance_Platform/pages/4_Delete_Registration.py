import streamlit as st 
from Home import face_rec
import cv2 
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title='Delete Registration',layout='centered')

st.subheader("Delete Registration")

## Instantiate Registration Form
deletion_form = face_rec.RegistrationForm()

# Collect Person name and role
# Form

names_df = face_rec.retrieve_name(name='academy:register')

person_name = st.selectbox(label='Select their name', options=names_df)

role = st.selectbox(label='Select their Role', options=('Student','Teacher'))


# Delete the data in redis database

if st.button('Submit'):
  
  person_name_key = person_name+"@"+role
  
  
  return_val = deletion_form.delete_data_in_redis_db(person_name_key,role)
  
  if return_val == True:
    st.success(f"{person_name} deleted sucessfully")
    with st.spinner('Retrieving new registration from Redis DB ... '):
      redis_face_db = face_rec.retrieve_data(name='academy:register')
      st.dataframe(redis_face_db)
  else:
    st.error(f'{person_name} deletion was unsucessful, please indicate the correct role.')
    