from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.http import HttpResponse
import numpy as np
import urllib
import json
import cv2
import os
import tensorflow as tf

# @csrf_exempt
# def detect(request):
#     data = {"success": False}

#     if request.method == "POST":
#         data.update({"success": True})

#     return JsonResponse(data)

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

#         data.update({"num_digits": len(prediction_list), "success": True})
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



@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    # data = {"success": False, "method":"none"}
    JSONresponse = "{'status':'false'}"

    # if request.method == "GET":
    #     val = json.loads(request.body)['key']
    #     val = val + "yes"
    #     # data.update({"success":True, "method": "get", "JSONdata":val })
    #     data.update({"success":True, "method": "get2"})

    #     return JsonResponse(data)
        
    # check to see if this is a post request
    if request.method == "GET":

        if 'name' in request.GET:
            JSONresponse = "{'status':'true', 'field1': 'yes1'}"
        else:
            JSONresponse = "{'status':'true', 'field1': 'yes2'}"

        # sent_data = json.loads(request.body.decode("utf-8"))['name']

        # JSONresponse = "{'status':'true', 'data': {'name':'ralph', 'hobby':'tennis'}}"+ sent_data

    elif request.method == "POST":

        JSONresponse = "{'status':'true', 'data': {'name':'bob', 'hobby':'running' }}"

        # JSONdata = json.loads(request.body.decode("utf-8"))
        # val = json.loads(request.body)['key']
        # val = request.body['key']
        # body_unicode = request.body.decode('utf-8')
        # body = json.loads(body_unicode)
        # content = int(body['key']) +2

        # body = request.body # json object



        # body_unicode = request.body.decode('utf-8')
        # body = json.loads(body_unicode)
        # content = int(body['key']) +2


        # data.update({"success":True, "method": "post"})
        # data = "Succesfull"

    return HttpResponse(JSONresponse)

    # return JsonResponse(data)


    #     # received_json_data = json.loads(request.body.decode("utf-8"))
    #     # data.update({"success":True, "json": received_json_data})

    #     # name = request.POST.get("name")
    #     # lastname = request.POST.get("lastname")

    #     # d = request.POST

    #     # data.update({"success":True, "name":name, "lastname":lastname, "d":d  })

    #     # check to see if an image was uploaded
    #     if request.FILES.get("image", None) is not None:
    #         # grab the uploaded image
    #         image = _grab_image(stream=request.FILES["image"])
    #         # otherwise, assume that a URL was passed in
    #     else:
    #         # grab the URL from the request
    #         url = request.POST.get("url", None)
    #         # if the URL is None, then return an error
    #         if url is None:
    #             data["error"] = "No URL provided."
    #             return JsonResponse(data)
    #         # load the image and convert
    #         image = _grab_image(url=url)

    #     # storing upper left corner and down right corner
    #     coordinates = []

    #     interval = 1
    #     initial = 0
    #     final = initial+interval
    #     l = 0
    #     r = 0

    #     row, column = image.shape
    #     index = 0

    #     while (initial+interval < column):
    #         partial = image[:,initial:final]
    #         next_partial = image[:, final:final+interval]

    #         if (np.count_nonzero(partial == 1) == 0 and np.count_nonzero(next_partial == 1) != 0 ):
    #             temp_list = []
    #             l_x = final
    #             temp_list.append(l_x)
    #             coordinates.append(temp_list)
    #             l=l+1

    #         if ((np.count_nonzero(partial == 1) != 0 and np.count_nonzero(next_partial == 1) == 0) or (final == column-1 and r == l-1)):
    #             r_x = final
    #             coordinates[index].append(r_x)
    #             index = index + 1
    #             r=r+1

    #         # increase frame
    #         initial = initial+interval
    #         final = final+interval

    #     model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model/digit_detection')
    #     model = tf.keras.models.load_model(model_path)

    #     prediction_list = []

    #     for l_x,r_x in coordinates:
    #         temp_img = image[:,l_x:r_x]
    #         y,x = temp_img.shape

    #         to_add = int((64 - x)/2)
    #         diff = 64 - (x+ (to_add *2))
    #         to_add_left = to_add
    #         to_add_right = to_add + diff

    #         temp_img = np.append(temp_img, np.zeros((64,to_add_left )), axis=1)
    #         temp_img = np.append(np.zeros((64,to_add_right )),temp_img, axis=1)

    #         # prediction
    #         prediction_input = np.array([temp_img])
    #         prediction = model.predict(prediction_input)
    #         prediction = np.argmax(prediction)
    #         prediction_list.append(prediction)

    #     data.update({"num_digits": len(prediction_list), "success": True})
    #     for index in prediction_list:
    #         data.update({str(index): str(prediction_list[index])})

    # return JsonResponse(data)

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

