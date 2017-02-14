import argparse
import base64
import json
import os

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from datetime import datetime

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
# for recording
# record_folder = './debug2/1/'

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle_sim = data["steering_angle"]
    # The current throttle of the car
    throttle_sim = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    #image_array = image_array[55:121, 60:260, :]
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the
    # images. Feel free to change this.
    steering_angle = float(model.predict(
        transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free
    # to edit this.
    throttle = 0.3
    print(steering_angle, throttle, '%.3f' %
          (steering_angle * 25.05), steering_angle_sim)
    send_control(steering_angle, throttle)

    """
    for recording
    logpath = record_folder + 'driving_log.csv'
    logstring = ''
    file_path = 'IMG/center_' + \
        datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3] + '.jpg'
    image.save(record_folder + file_path)
    if os.path.isfile(logpath):
        logstring = ',%s,0,%s\n' % (throttle_sim, speed)
    logstring += file_path + ',,,%s' % str(steering_angle)

    with open(logpath, 'a') as logfile:
        logfile.write(logstring)
    """

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    """
    for recording
    if not os.path.exists(record_folder):
        os.makedirs(record_folder)

    if not os.path.exists(record_folder + 'IMG'):
        os.makedirs(record_folder + 'IMG')
    """
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        # model = model_from_json(jfile.read())
        model = model_from_json(json.loads(jfile.read()))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
