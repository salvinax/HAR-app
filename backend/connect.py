import requests
import time
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from multiprocessing import Pool, Process, Queue
import pickle
import matplotlib.pyplot as plt
import asyncio
import websockets
import json
import os
from io import BytesIO
import base64

local = "http://192.168.2.48/get"
dataNames = ["accX", "accY", "accZ"]
url = local + "?" + ("&".join(dataNames))
timer = 0
graphX, graphY, graphZ = [], [], []
results = []
sampling_interval = 20
time_interval = 0.1
sample_seconds = 2
sampling_queue = Queue()



# apply segmentation with window size
# set window to be 2 seconds with a sampling rate is 10/sec (2*10 = 20 samples)
def read_accelerometer_data(sampling_queue):
    while True: 
        time.sleep(time_interval)
        try: 
            req = requests.get(url=url, timeout = 3)
      
            if req.status_code == 200: 
                data = req.json()
                x = data['buffer']['accX']['buffer'][0]
                y = data['buffer']['accY']['buffer'][0]
                z = data['buffer']['accZ']['buffer'][0]
                if x: 
                    sampling_queue.put([x, y, z])
            else: 
                print(f"Request failed with status code {data.status_code}")
        except requests.RequestException as e: 
             print(f"An error occured: {e}")
     

def findfeatures(window):
        maxim = np.max(window)
        minim = np.min(window)
        range_win = maxim - minim
        mean = np.mean(window)
        median = np.median(window)
        std = np.std(window)
        skew_win = skew(window)
        kurt_win = kurtosis(window)
        variance = np.var(window)
        rms_win = np.sqrt(sum([value**2 for value in window]))
        return [maxim, minim, range_win, mean, median, std, skew_win, kurt_win, variance, rms_win]


def extractFeatures(data): 
    global graphX
    global graphY
    global graphZ
    x_window, y_window, z_window = zip(*data)
    x_window = list(x_window)
    y_window = list(y_window)
    z_window = list(z_window)

    
    graphX += x_window
    graphY += y_window
    graphZ += z_window

    x_features = findfeatures(x_window)
    y_features = findfeatures(y_window)
    z_features = findfeatures(z_window) 

    #find correlation between axis data
    coeffxy = np.corrcoef(x_window, y_window)[0, 1]
    coeffxz = np.corrcoef(x_window, z_window)[0, 1]
    coeffyz = np.corrcoef(y_window, z_window)[0, 1]

    window_features = x_features + y_features + z_features

    # create dataframe
    features = pd.DataFrame([window_features], columns=['x_max', 'y_max', 'z_max','x_min', 'y_min', 'z_min','x_range', 'y_range', 'z_range','x_mean', 'y_mean', 'z_mean',
                                                'x_median', 'y_median', 'z_median','x_std', 'y_std', 'z_std', 'x_skew', 'y_skew', 'z_skew','x_kurt', 'y_kurt', 'z_kurt',
                                                'x_variance', 'y_variance', 'z_variance', 'x_rms','y_rms', 'z_rms'])
    
    features['coeffxy'] = coeffxy
    features['coeffyz'] =  coeffyz
    features['coeffxz'] = coeffxz 

    model_path = 'model.pkl'
    model = pickle.load(open(model_path, 'rb'))
    prediction = model.predict(features)
    results.append(prediction[0])
    return prediction[0]

# graph simple 0/1 and xyz plot with matplotlib
def graph():

    fig, ax = plt.subplots(2)
    time1 = np.arange(0, len(results)*sample_seconds, sample_seconds)
    ax[0].set_facecolor('#1a1a1a')
    fig.set_facecolor('#1a1a1a')

    font_properties = {
    'family': "sans-serif",  # Change to the desired font family
    'color': "white"
    }


   
    ax[0].step(time1, results, color='#FFFFFF')
    ax[0].set_ylabel('Predicted Activity', **font_properties)
    ax[0].set_yticks([0, 1])
    ax[0].set_yticklabels(['walk', 'jump'], color='white')
    ax[0].set_ylim(-0.1, 1.1)


    # Set the font color of labels to white
    ax[0].xaxis.label.set_color('white')
    ax[0].yaxis.label.set_color('white')

    # Set the font color of tick labels to white
    ax[0].tick_params(axis='x', colors='white')
    ax[0].tick_params(axis='y', colors='white')

     # Set the font color of labels to white
    ax[1].xaxis.label.set_color('white')
    ax[1].yaxis.label.set_color('white')

    # Set the font color of tick labels to white
    ax[1].tick_params(axis='x', colors='white')
    ax[1].tick_params(axis='y', colors='white')

    ax[0].spines['bottom'].set_color('white')
    ax[0].spines['left'].set_color('white')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    
    ax[1].spines['bottom'].set_color('white')
    ax[1].spines['left'].set_color('white')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

  
    time2 = np.arange(0, len(results)*sample_seconds, time_interval)
    ax[1].set_facecolor('#1a1a1a')
    ax[1].plot(time2, graphX, label='x', linewidth=1)
    ax[1].plot(time2, graphY, label='y', linewidth=1)
    ax[1].plot(time2, graphZ, label='z', linewidth=1)
    ax[1].set_ylabel('Acceleration (m/s^2)', **font_properties)
    ax[1].set_xlabel('Time (s)', **font_properties)
    ax[1].legend() 
    plt.subplots_adjust( wspace=0.4,hspace=0.4)

    # cwd = os.getcwd()
    # plt.savefig(os.path.join(cwd, 'plot.png'))

    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')

    return encoded_image


async def handler(websocket):
     # Start the data collection process
    data_collection_process = Process(target=read_accelerometer_data, args=(sampling_queue,))
    data_collection_process.start()

    data_buffer = []

    # Start the preprocessing and prediction process
    #processing_task = asyncio.create_task(predict(sampling_queue, websocket))

    try:
        while True:
            sensor_data = sampling_queue.get()
            data_buffer.append(sensor_data)

            if len(data_buffer) >= sampling_interval:
                graph()
                result = extractFeatures(data_buffer[:sampling_interval])
                print(result)

                data_buffer = data_buffer[sampling_interval:]
                await websocket.send(json.dumps({'type': 'pred', 'result': int(result)}))
                encoded_img = graph()
                await websocket.send(json.dumps({'type': 'image', 'img': encoded_img}))
            
            await asyncio.sleep(0)
    except websockets.exceptions.ConnectionClosedError:
        # Connection closed by the client
        data_collection_process.terminate()
        #processing_task.cancel()



async def main():
    async with websockets.serve(handler, "", 8001): # starting websocket server
        await asyncio.Future()

if __name__ == '__main__':
     asyncio.run(main())