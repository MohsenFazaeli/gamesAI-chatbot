from flask import Flask, render_template, jsonify, request

##AI Loader
from applications.aiEngine.test.Tester import Tester

from applications.aiEngine.dataGen.questions_gen import QUESTION_GEN
from applications.aiEngine.dataGen.questions_db_util import QDB
q_gen = QUESTION_GEN(QDB())
q_gen.generate()

from applications.aiEngine.test.Tester import Tester
from applications.aiEngine.model.JointIntentAndSlotFillingModel import JointIntentAndSlotFillingModel as JM
from transformers import AutoModel as OurModel
from shared.constants.constants import model_name, hf_path, xlm_mixed_path

xlmr = None
try:
    xlmr = OurModel.from_pretrained(xlm_mixed_path)
    print("Base Model loaded as it should")
except Exception as e:
    print(f"Base path {xlm_mixed_path} is not valid", e)
#print(xlmr)

if xlmr is None or True:
    try:
        xlmr = OurModel.from_pretrained(hf_path)
    except Exception as e:
        print(f"Base path {hf_path} is not valid", e)
        xlmr = OurModel.from_pretrained(model_name)
    #xlmr.save_pretrained()
else:
    pass
    # print("Model loaded as it should")

model = JM(bert=xlmr, method='Thick')
model.load_and_save_pretrained_model()
print("Model is Pretrained: ", model.pre_trained_model)

tester = Tester(q_gen.df, model)

from applications.db.api_provider import *




app = Flask(__name__)

# Define a function to add CORS headers to every response
def add_cors_headers(response):
    # Allow requests from any origin
    response.headers['Access-Control-Allow-Origin'] = '*'
    # Allow the GET, POST, and OPTIONS methods
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    # Allow the Content-Type header
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Apply the CORS headers to all responses using the after_request decorator
@app.after_request
def after_request(response):
    return add_cors_headers(response)


@app.route('/hw')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/api/bot-response')
def bot():  # put application's code here
    # Your API logic here
    user_message = request.args.get('user_message')
    print(user_message)

    processed = tester.nlu(user_message)
    print(processed)

    api = QueryAPI()
    respo = api.query(processed)
    print(respo)

    data = {'param1': respo}
    return jsonify(data)

# Route for serving the HTML page
@app.route('/')
def index():
    return render_template('index.html')


# Route for the REST API
@app.route('/api/data', methods=['GET'])
def api_data():
    # Your API logic here
    data = {'key': 'value'}
    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
