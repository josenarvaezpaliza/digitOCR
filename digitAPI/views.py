from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.http import HttpResponse
import numpy as np
import urllib
import json
import cv2
import os
import base64
import io
import tensorflow as tf


# @csrf_exempt
# def detect(request):
#     # initialize the data dictionary to be returned by the request
#     data = {"success": False}

#     # check to see if this is a post request
#     if request.method == "POST":
#         # check to see if an image was uploaded
#         if request.FILES.get("image", None) is not None:
#             # grab the uploaded image
#             image = _grab_image(stream=request.FILES["image"])
#             # otherwise, assume that a URL was passed in
#         else:
#             # grab the URL from the request
#             url = request.POST.get("url", None)
#             # if the URL is None, then return an error
#             if url is None:
#                 data["error"] = "No URL provided."
#                 return JsonResponse(data)
#             # load the image and convert
#             image = _grab_image(url=url)

#         # storing upper left corner and down right corner
#         coordinates = []

#         interval = 1
#         initial = 0
#         final = initial+interval
#         l = 0
#         r = 0

#         row, column = image.shape
#         index = 0

#         while (initial+interval < column):
#             partial = image[:,initial:final]
#             next_partial = image[:, final:final+interval]

#             if (np.count_nonzero(partial == 1) == 0 and np.count_nonzero(next_partial == 1) != 0 ):
#                 temp_list = []
#                 l_x = final
#                 temp_list.append(l_x)
#                 coordinates.append(temp_list)
#                 l=l+1

#             if ((np.count_nonzero(partial == 1) != 0 and np.count_nonzero(next_partial == 1) == 0) or (final == column-1 and r == l-1)):
#                 r_x = final
#                 coordinates[index].append(r_x)
#                 index = index + 1
#                 r=r+1

#             # increase frame
#             initial = initial+interval
#             final = final+interval

#         model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model/digit_detection')
#         model = tf.keras.models.load_model(model_path)

#         prediction_list = []

#         for l_x,r_x in coordinates:
#             temp_img = image[:,l_x:r_x]
#             y,x = temp_img.shape

#             to_add = int((64 - x)/2)
#             diff = 64 - (x+ (to_add *2))
#             to_add_left = to_add
#             to_add_right = to_add + diff

#             temp_img = np.append(temp_img, np.zeros((64,to_add_left )), axis=1)
#             temp_img = np.append(np.zeros((64,to_add_right )),temp_img, axis=1)

#             # prediction
#             prediction_input = np.array([temp_img])
#             prediction = model.predict(prediction_input)
#             prediction = np.argmax(prediction)
#             prediction_list.append(prediction)

#         data.update({"num_digits": len(prediction_list), "success": coordinates})
#         for index in prediction_list:
#             data.update({str(index): str(prediction_list[index])})

#     return JsonResponse(data)

# def _grab_image(path=None, stream=None, url=None):
#     # if the path is not None, then load the image from disk
#     if path is not None:
#         image = imread(path, cv2.IMREAD_GRAYSCALE)
#         image = process_img(image)
        
#     # otherwise, the image does not reside on disk
#     else:	
#         # if the URL is not None, then download the image
#         if url is not None:
#             resp = urllib.urlopen(url)
#             data = resp.read()
#         # if the stream is not None, then the image has been uploaded
#         elif stream is not None:
#             data = stream.read()
#         # convert the image to a NumPy array and then read it into
#         # OpenCV format
#         image = np.asarray(bytearray(data), dtype="uint8")
#         image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
#         image = process_img(image)
#     # return the image
#     return image

# # Create your views here.

# def process_img(image):
#     img = cv2.resize(image,(64,64))
#     thresh = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     thresh = thresh/255.0
#     thresh = thresh.astype(int)
#     #invert colours
#     if (np.bincount(thresh.flatten()).argmax()) == 1:
#         thresh = 1-thresh

#     return thresh



# @csrf_exempt
# def detect(request):
#     # initialize the data dictionary to be returned by the request
#     data = {"success": False}

#     # check to see if this is a post request
#     if request.method == "POST":
        
#         try:
#             #GETTING MODEL
#             model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model/digit_detection_mnist6')
#             model = tf.keras.models.load_model(model_path)

#             #GETTING IMAGE FROM POST DATA
#             body = json.loads(request.body.decode("utf-8"))

#             base64_string = body['image']
#             decoded_data = base64.b64decode(base64_string)
#             np_data = np.frombuffer(decoded_data,np.uint8)
#             # Grayscale image
#             image = cv2.imdecode(np_data,cv2.IMREAD_GRAYSCALE)
#             data.update({"success": True ,"size": image.shape})

#             prediction_count = 1
#             #crop image according to bounding boxes
#             bounding_boxes = body['boxes']
#             for box in bounding_boxes:
#                 x1 = int(box[0])
#                 y1 = int(box[1])
#                 x2 = int(box[2])
#                 y2 = int(box[3])
#                 # data.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

#                 temp_img = image[y1:y2, x1:x2]
#                 # data.update({"success": True ,"size2": temp_img.shape})

#                 prediction = predict_digits(temp_img, model)
#                 prediction_string = "prediction"+str(prediction_count)
#                 data.update({"'"+prediction_string+"'": prediction})
#                 prediction_count = prediction_count+1

#             # #PREDICTION
#             # # prediction = predict_digits(image, model)

#             # #UPDATE DATA
#             # data.update({"success": True ,"prediction": prediction})
#         except:
#             data.update({"success": True ,"prediction": "ERROR"})

#     return JsonResponse(data)

# def predict_digit(img, model):
#     prediction_input = np.array([img])
#     prediction = model.predict(prediction_input)
#     prediction = np.argmax(prediction)
#     return prediction

# def pre_process(img):
#     # img = img[1760:1960,1315:1580]
#     # img = img[2935:3180, 2500:2865]
#     # img = img[ 2980:3210, 365:870]
#     thresh = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     thresh = thresh/255.0
#     thresh = thresh.astype(int)
#     #invert colours
#     if (np.bincount(thresh.flatten()).argmax()) == 1:
#         thresh = 1-thresh
    
#     return thresh

# def extract_digits(img):
#     # storing upper left corner and down right corner
#     coordinates = []

#     # setup
#     interval = 1
#     initial = 0
#     l = 0
#     r = 0

#     row, column = img.shape
#     index = 0

#     coordinate = []

#     while (initial+interval < column):
#         current_frame = img[:,initial:initial+interval]
#         next_frame = img[:, initial+interval:initial+interval+interval]

#         if (np.count_nonzero(current_frame == 1) == 0 and np.count_nonzero(next_frame == 1) != 0 and l == r ):
#             temp_list = []
#             l_x = initial+interval
#             coordinate.append(l_x)
#             l=l+1
            

#         if ((np.count_nonzero(current_frame == 1) != 0 and np.count_nonzero(next_frame == 1) == 0 and r == l-1 ) or (initial+interval == column-interval and r == l-1 )):
#             r_x = initial+interval
#             coordinate.append(r_x)
#             coordinates.append(coordinate)
#             coordinate = []
#             r=r+1

#         # increase frame
#         initial = initial+interval

#     return coordinates

# def predict_digits(img, model):

#     img = pre_process(img)
#     coordinates = extract_digits(img)
    
#     prediction_list = []
    
#     for coordinate in coordinates:
#         l_x = coordinate[0]
#         r_x = coordinate[1]

#         if (r_x - l_x < 10):
#             continue

#         temp_img = img[:,l_x:r_x]
#         y,x = temp_img.shape

#         # add half and half
#         to_add = int((y - x)/2)
#         diff = y - (x+ (to_add *2))
#         to_add_left = to_add
#         to_add_right = to_add + diff
        
#         temp_img = np.append(temp_img, np.zeros((y,to_add_left )), axis=1)
#         temp_img = np.append(np.zeros((y,to_add_right )),temp_img, axis=1)

#         temp_img = cv2.resize(temp_img,(64,64))
#         thresh_1 = temp_img > 0.3
#         thresh_2 = temp_img <= 0.3
#         temp_img[thresh_1] = 1
#         temp_img[thresh_2] = 0
        
#         temp_img = np.dstack(temp_img).T

#         prediction = predict_digit(temp_img, model)
#         prediction_list.append(str(prediction))

#     return prediction_list



# def _grab_image(path=None, stream=None, url=None):
#     # if the path is not None, then load the image from disk
#     if path is not None:
#         image = imread(path, cv2.IMREAD_GRAYSCALE)
#         # image = process_img(image)
        
#     # otherwise, the image does not reside on disk
#     else:	
#         # if the URL is not None, then download the image
#         if url is not None:
#             resp = urllib.urlopen(url)
#             data = resp.read()
#         # if the stream is not None, then the image has been uploaded
#         elif stream is not None:
#             data = stream.read()
#         # convert the image to a NumPy array and then read it into
#         # OpenCV format
#         image = np.asarray(bytearray(data), dtype="uint8")
#         image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
#         # image = process_img(image)
#     # return the image
#     return image



@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        
        try:
            #GETTING MODEL
            model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model/digit_detection_mnist6')
            model = tf.keras.models.load_model(model_path)

            #GETTING IMAGE FROM POST DATA
            body = json.loads(request.body.decode("utf-8"))

            base64_string = body['image']
            decoded_data = base64.b64decode(base64_string)
            np_data = np.frombuffer(decoded_data,np.uint8)
            # Grayscale image
            image = cv2.imdecode(np_data,cv2.IMREAD_GRAYSCALE)
            data.update({"success": True ,"size": image.shape})

            prediction_count = 1
            #crop image according to bounding boxes
            bounding_boxes = body['boxes']
            for box in bounding_boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                # data.update({"'"+"x1"+str(prediction_count)+"'": x1, "'"+"y1"+str(prediction_count)+"'": y1, "'"+"x2"+str(prediction_count)+"'": x2, "'"+"y2"+str(prediction_count)+"'": y2})

                temp_img = image[y1:y2, x1:x2]
                # data.update({"success": True ,"size2": temp_img.shape})

                prediction = predict_digits(temp_img, model)
                prediction_string = "prediction"+str(prediction_count)
                data.update({"'"+prediction_string+"'": prediction})
                prediction_count = prediction_count+1

            # #PREDICTION
            # # prediction = predict_digits(image, model)

            # #UPDATE DATA
            # data.update({"success": True ,"prediction": prediction})
        except:
            data.update({"success": True ,"prediction": "ERROR"})

    return JsonResponse(data)

def predict_digit(img, model):
    prediction_input = np.array([img])
    prediction = model.predict(prediction_input)
    prediction = np.argmax(prediction)
    return prediction

def pre_process(img):
    thresh = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = thresh/255.0
    thresh = thresh.astype(int)
    #invert colours
    if (np.bincount(thresh.flatten()).argmax()) == 1:
        thresh = 1-thresh
    
    return thresh

def extract_digits(img):
    # storing upper left corner and down right corner
    coordinates = []

    # setup
    interval = 1
    initial = 0
    l = 0
    r = 0

    row, column = img.shape
    index = 0

    coordinate = []

    while (initial+interval < column):
        current_frame = img[:,initial:initial+interval]
        next_frame = img[:, initial+interval:initial+interval+interval]

        if (np.count_nonzero(current_frame == 1) == 0 and np.count_nonzero(next_frame == 1) != 0 and l == r ):
            temp_list = []
            l_x = initial+interval
            coordinate.append(l_x)
            l=l+1
            

        if ((np.count_nonzero(current_frame == 1) != 0 and np.count_nonzero(next_frame == 1) == 0 and r == l-1 ) or (initial+interval == column-interval and r == l-1 )):
            r_x = initial+interval
            coordinate.append(r_x)
            coordinates.append(coordinate)
            coordinate = []
            r=r+1

        # increase frame
        initial = initial+interval

    return coordinates

def predict_digits(img, model):

    img = pre_process(img)
    coordinates = extract_digits(img)
    
    # temp_img = cv2.resize(img,(64,64))

    # temp_img = np.dstack(temp_img).T

    # prediction = predict_digit(temp_img, model)
    prediction_list = []
    # prediction_list.append(str(prediction))
    
    for coordinate in coordinates:
        l_x = coordinate[0]
        r_x = coordinate[1]

        if (r_x - l_x < 10):
            continue

        temp_img = img[:,l_x:r_x]
        y,x = temp_img.shape

        # add half and half
        to_add = int((y - x)/2)
        diff = y - (x+ (to_add *2))
        to_add_left = to_add
        to_add_right = to_add + diff
        
        temp_img = np.append(temp_img, np.zeros((y,to_add_left )), axis=1)
        temp_img = np.append(np.zeros((y,to_add_right )),temp_img, axis=1)

        temp_img = cv2.resize(temp_img,(64,64))
        thresh_1 = temp_img > 0.3
        thresh_2 = temp_img <= 0.3
        temp_img[thresh_1] = 1
        temp_img[thresh_2] = 0
        
        temp_img = np.dstack(temp_img).T

        prediction = predict_digit(temp_img, model)
        prediction_list.append(str(prediction))

    return prediction_list



def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = imread(path, cv2.IMREAD_GRAYSCALE)
        # image = process_img(image)
        
    # otherwise, the image does not reside on disk
    else:	
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.urlopen(url)
            data = resp.read()
        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()
        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        # image = process_img(image)
    # return the image
    return image









