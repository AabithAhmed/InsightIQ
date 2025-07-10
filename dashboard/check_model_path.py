import os

print("Current working directory:", os.getcwd())
print("Model file exists:", os.path.exists('../models/churn_model.pkl'))
print("Absolute path:", os.path.abspath('../models/churn_model.pkl'))
