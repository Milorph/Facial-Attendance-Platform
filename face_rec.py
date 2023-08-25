import numpy as np
import pandas as pd
import cv2
import os
import redis
import logging

# Insight face

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Date time

import time
from datetime import datetime

# Connect to redis client
hostname = 'redis-17742.c11.us-east-1-2.ec2.cloud.redislabs.com'
portnumber = 17742
password = 'l8PAQHWT9eLqCDXW0UAtdEtwwy3cuqIe'

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)

# Retrieve Data from Database
def retrieve_data(name):
    retrieve_dict = r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(), index))
    retrieve_series.index = index
    retrieve_df = retrieve_series.to_frame().reset_index()
    retrieve_df.columns = ['name_role','facial_features']
    retrieve_df[['Name','Role']] = retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieve_df[['Name','Role','facial_features']]

def retrieve_name(name):
    retrieve_dict = r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(), index))
    retrieve_series.index = index
    retrieve_df = retrieve_series.to_frame().reset_index()
    retrieve_df.columns = ['name_role','facial_features']
    retrieve_df[['Name','Role']] = retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return list(retrieve_df['Name'])

# Configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc', roo ='insightFace_model', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe,feature_column,test_vector,name_role=['Name','Role'],thresh=0.5):

    # Take the dataframe
    dataframe = dataframe.copy()
    # Index face embedding from the dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)

    # Call cosine similarity
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # Filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]

    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name, person_role

# Real Time Prediction
# Save Logs for every 1 min

class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[],role=[],current_time=[])
    
    def reset_dict(self):
        self.logs = dict(name=[],role=[],current_time=[])
    
    def saveLogs_redis(self):
        # Create a logs dataframe
        dataframe = pd.DataFrame(self.logs)
        # Drop the duplicate information (distinct name)
        dataframe.drop_duplicates('name',inplace=True)
        # Push data to redis database
        # encode the data
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data =[]
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data) > 0:
            r.lpush("attendance:logs", *encoded_data)
        
        self.reset_dict()
        
    def face_prediction(self, test_image,dataframe,feature_column,name_role=['Name','Role'],thresh=0.5):
        
        # Finding the time
        current_time = str(datetime.now())
        
        # Take the test image and apply to insight face
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        
        # Use a for loop and extract each embedding and pass to ml_search_algorithm
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,
                                                        feature_column,
                                                        test_vector = embeddings,
                                                        name_role=name_role,
                                                        thresh=thresh)
        
            if person_name == 'Unknown':
                color = (0,0,255)
            else:
                color = (0,255,0)
                
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
        
            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)
            cv2.putText(test_copy, current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)
            # Save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
            
        return test_copy
    
#### Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
        
    def reset(self):
        self.sample = 0
        
    def get_embedding(self, frame):

        results = faceapp.get(frame,max_num=1) # SELECT 1 PARTICULAR FACE
        
        embeddings = None
        
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)

            # Put text samples info
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            
            # facial features
            embeddings = res['embedding']
        
        return frame, embeddings

    def save_data_in_redis_db(self,name,role):
        # Validation name
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'name_false'

            # if face_embedding.txt exists
            if 'face_embedding.txt' not in os.listdir():
                return 'file_false'
        
        # Load face_embedding.txt
        x_array = np.loadtxt('face_embedding.txt',dtype=np.float32) # flatten array
        
        # Convert into array (proper shape)
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples,512)
        x_array = np.asarray(x_array)
        
        # Call the mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()
        
        # Save them into redis database
        # redis hashes
        r.hset(name='academy:register',key=key,value=x_mean_bytes)
        
        os.remove('face_embedding.txt')
        self.reset()
        
        return True
    
    def delete_data_in_redis_db(self, name, role):
        try:
            # Validation name
            if role is not None and name is not None:
                    key = name
                    if r.hexists('academy:register', key):
                        r.hdel('academy:register', key)
                        logging.info(f"Deleted {key} from academy:register")
                        return True
                    else:
                        logging.warning(f"{key} not found in academy:register")
                        return False
            else:
                logging.warning("Empty name provided")
                return False
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
        
        return False
    
        
    
