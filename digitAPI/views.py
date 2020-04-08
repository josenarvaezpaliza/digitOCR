from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

@csrf_exempt
def detect(request):
    data = {"success": False}

    if request.method == "POST":
        data.update({"success": True})

    return JsonResponse(data)

