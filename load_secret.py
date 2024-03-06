import json
import os

def load_secret():

    secret = json.load(open('kaggle.json'))

    os.environ["KAGGLE_USERNAME"] = secret['username']
    os.environ["KAGGLE_KEY"] = secret['key']
