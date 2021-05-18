# Import libraries
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import json
from Recommender import recommend

# Creat object for Flask App
app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Welcome to AmazeBasket Ingredient Recommendation API"


@app.route("/<item>", methods=['POST', 'GET'])
def index(item):
    if request.method == "POST":
        recommendation = recommend(item)
        return recommendation
    else:
        return 'Welcome to AmazeBasket Ingredient Recommendation API'


# Main Function
if __name__ == '__main__':
    app.run(debug=True)
