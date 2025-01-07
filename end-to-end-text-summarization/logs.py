from Summarizer.logger import logging
from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    logging.info("We are testing our logging file")
    
    return "Hello worlds"

if __name__=="__main__":
    app.run(debug=True) # port: 5000