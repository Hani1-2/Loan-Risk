from flask import Flask, render_template, request
import pickle
import numpy as np
model = pickle.load(open('loan.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    arr = []
    if request.method == 'POST':
        data1 = request.form['a']
        if data1 == "1":
            data1 = 1
        else:
            data1 = 0
        arr.append(data1)
        data2 = request.form['b']
        if (data2 == "Yes"):
            data2 = 1
        else:
            data2 = 0
        arr.append(data2)
        data3 = request.form['c']
        if (data3 == "0"):
            data3 = 0
        elif (data3 == "1"):
            data3 = 1
        elif (data3 == "2"):
            data3 = 2
        else:
            data3 = 3
        arr.append(data3)
        # data3 means Nature
        data4 = request.form['d']
        if (data4 == "Graduate"):
            data4 = 0
        else:
            data4 = 1
        arr.append(data4)
    # data5 means Mental Peace
        data5 = request.form['e']
        if data5 == "Yes":
            data5 = 1
        else:
            data5 = 0
        arr.append(data5)
    # data6 means Reaction on lack  of somthing
        data6 = request.form['f']
        if type(data6) == int or type(data6) == float:
            arr.append(data6)
        else:
            arr.append(0)
        data7 = request.form['g']
        if type(data7) == int or type(data6) == float:
            arr.append(data7)
        else:
            arr.append(0)
        data8 = request.form['h']
        if type(data8) == int or type(data6) == float:
            arr.append(data8)
        else:
            arr.append(0)
        data9 = request.form['i']
        arr.append(data9)
        # if type(data9) == int or type(data6) == float:
        #     arr.append(data9)
        # else:
        #     arr.append(0)
        data10 = request.form['j']
        if data10 == "1":
            data10 = 1
        else:
            data10 = 0
        arr.append(data10)
        arr1 = np.array([arr])
        arr2 = arr1.astype(np.float64)
        pred = model.predict(arr2).reshape(1, -1)
        print(pred)
        return render_template("result.html", data=pred)


if __name__ == "__main__":
    app.run(debug=True)
