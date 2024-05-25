from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__, static_folder=os.path.abspath('../client/dist'), static_url_path='')

# Load the pretrained model
model = joblib.load('pipe.pkl')

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive input from the frontend
    
    Company = request.form['brand']
    TypeName = request.form['type']
    Ram = int(request.form['ram'])
    Weight = float(request.form['weight'])
    resolution = request.form['resolution']
    size = float(request.form['screen_size'])
    Cpu = request.form['processor']
    HDD = int(request.form['hdd'])
    SSD = int(request.form['ssd'])
    gpu_brand = request.form['gpuBrand']
    os = request.form['os']
    
    if (request.form['touch'] == "yes"):
        Touchscreen = 1
    else:
        Touchscreen = 0
    if (request.form['ips'] == "yes"):
        ips = 1
    else:
        ips = 0
        
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/size
    
    # Example: Use the input data to form input features
    import numpy as np
    query = np.array([Company, TypeName, Ram, Weight, Touchscreen, ips, ppi, Cpu, HDD, SSD, gpu_brand, os]).reshape(1, 12)
    
    # Make predictions
    
    # prediction = model.predict(query)
    prediction = str(int(np.exp(model.predict(query)[0])))
    # Return the result to the frontend
    # print("Predicted value:", str(int(np.exp(model.predict(query)[0]))))
    # print(prediction)
    # return render_template('index.html', prediction=prediction)
    return prediction
    

if __name__ == '__main__':
    app.run(debug=True)
