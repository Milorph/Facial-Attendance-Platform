import streamlit as st 
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure


st.set_page_config(page_title="Predictions",layout='centered')
st.subheader("Real-Time Attendance System")

# Retrieve the data from Redis Database
with st.spinner('Retrieving Data from Redis DB ... '):
  redis_face_db = face_rec.retrieve_data(name='academy:register')
  st.dataframe(redis_face_db)

st.success("Data sucessfully received from Redis DB")


# time
waitTime = 5 # time in seconds
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
      
      webrtc_streamer.stop(key="realtimePrediction")

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)