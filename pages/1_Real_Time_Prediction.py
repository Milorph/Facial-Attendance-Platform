import streamlit as st 
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time
import os

from twilio.rest import Client

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = os.environ['AC022c4944e9902bf04423ae08f4703c3e']
auth_token = os.environ['a44bcaf3cf5c9aa64566730ffd6e2656']
client = Client(account_sid, auth_token)

token = client.tokens.create()

st.set_page_config(page_title="Predictions",layout='centered')
st.subheader("Real-Time Attendance System")

# Retrieve the data from Redis Database
with st.spinner('Retrieving Data from Redis DB ... '):
  redis_face_db = face_rec.retrieve_data(name='academy:register')
  st.dataframe(redis_face_db)

st.success("Data sucessfully received from Redis DB")


# time
waitTime = 10 # time in seconds
setTime = time.time()
realtimepred = face_rec.RealTimePred() # Instantiate the class

#Real Time Prediction
# streamlit web-rtc

# callback function
def video_frame_callback(frame):
    global setTime
    
    img = frame.to_ndarray(format="bgr24")
    
    pred_img = realtimepred.face_prediction(img, 
                                        redis_face_db, 
                                        'facial_features',
                                        [
                                          'Name',
                                          'Role'
                                        ],
                                       thresh=0.5)

    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
      realtimepred.saveLogs_redis()
      setTime = time.time() # Reset time
      
      print('Save Data to redis')

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)