import flask
import pandas as pd
import io
from flask import request, jsonify, render_template, send_from_directory
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
####################################################
# Flask Config
app = flask.Flask(__name__)
app.config["DEBUG"] = True
####################################################

####################################################
# This block is used to initialize and read the training engine to be able to consume it in the api
# it will be initialized only once on app run, this way we don't have to train the engine on every api request.
testData = pd.read_csv("./assets/RBF_SVM.csv")
X = testData.drop('PCOS', axis=1)
y = testData['PCOS']
X_train, X_test, y_train, y_test = train_test_split(X, y)
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
####################################################


####################################################
# This Block is used to define the routing formula of the frontend
# Main Login 
@app.route('/', methods=['GET'])
def login_index():
    return render_template("index.html")
  
 # To Load index CSS 
@app.route("/index.css")
def index_style():
    return send_from_directory("templates", "index.css")

 # Form Page 
@app.route('/prediction-form', methods=['GET'])
def prediction_form():
    return render_template("form.html")

 # To Load form CSS 
@app.route("/form.css")
def form_style():
    return send_from_directory("templates", "form.css")
####################################################

####################################################
# This block is used to define the API's that will be needed to calculate the predictions, and recieve data

# A route to return all of the data used for training
@app.route('/api/ai/getTest', methods=['GET'])
def get_training_data():
    return testData.to_json()

# This API is used to compute the prediction according to the parameters sent from the frontend
@app.route('/api/ai/predict-result', methods=['POST'])
def compute_predict():
  try:
    # consume the request and parse it to a data frame object to be predicted
   df = pd.json_normalize(request.get_json())
   print('Model to Predict: ', request.get_json())
   y_pred = svclassifier.predict(df)
   print('Result came back as: ', y_pred)
   if y_pred == [1]: # if result is postive we return a success message for the user
    return 'Positive Result'
   elif y_pred == [0]: # if result is negative we return a negative message for the user
    return 'Negative Result'
   else:
    return 'Result was inconclusive' # if the prediction didn't work, we return inconclusive for the user
  except:
   return 'Result was inconclusive'  # in case of any general error, we return inconclusive for the user
####################################################

# to run the app and define which port  
app.run(debug=True, port=5000)