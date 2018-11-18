import requests
import json

class Toximeter():
    def __init__(self,endpoint='http://18.222.175.231:8000'):
        self.endpoint = endpoint
    def analyze(self,text):
        r = requests.get(self.endpoint+'/?text='+text)
        return r.json()
