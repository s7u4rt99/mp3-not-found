from flask import Flask, request
import db
import json
from flask_pymongo import pymongo

app = Flask(__name__)

CONNECTION_STRING = "mongodb+srv://cs3237:giveusaplus@cs3237.tpbbgzs.mongodb.net/?retryWrites=true&w=majority"

client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('cs3237')

# collection for incorrect predictions
incorrect_predictions = pymongo.collection.Collection(db, 'incorrect-predictions')

# collection for words
words = pymongo.collection.Collection(db, 'words')


@app.route('/hello', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"

#insert data to the data base
@app.route("/add/misspelled", methods=["POST"])
def addMisspelled():
    print(request)
    data = request.get_json()
    print(data)
    incorrect_predictions.insert_one(data)
    return "Added predicted: " + data["predicted"] + " autocorrected: " + data["autocorrected"] 

@app.route("/add/word", methods=["POST"])
def addWord():
    print(request)
    data = request.get_json()
    print(data)
    words.insert_one(data)
    return "Added word: " + data["word"]

if __name__ == '__main__':
    app.run(port=8000)