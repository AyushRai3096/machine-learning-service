from django.shortcuts import render
from mlservice.Classifier import VoteClassifier

from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

# Create your views here.
@api_view(['POST',])
def predict(request):

    if 'sentence' not in request.POST:
        msg = {
            "error": "Please provide a sentence"
        }
        return Response(msg, status=status.HTTP_400_BAD_REQUEST)

    sentence = request.POST["sentence"]
    classifier = VoteClassifier()

    sentiment = classifier.get_sentiment(sentence)
    response = {
        "sentiment": sentiment
    }

    return Response(response)


@api_view(['POST',])
def keywords(request):

    if 'sentence' not in request.POST:
        msg = {
            "error": "Please provide a sentence"
        }
        return Response(msg, status=status.HTTP_400_BAD_REQUEST)

    sentence = request.POST["sentence"]
    classifier = VoteClassifier()
    keywords_dict = classifier.find_features(sentence)

    res = []
    for key in keywords_dict:
        if keywords_dict[key]:
            res.append(key)
    
    response = {
        "keywords" : res
    }

    return Response(response)


