from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn import linear_model
import scipy.stats as stat
class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):#,**kwargs):    # we inherit everything from the original LR class
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        
        #### Get multivariate p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values   # p-values are stored here




model = pickle.load(open('approved_or_not.pkl', 'rb'))

app = Flask(__name__)

@app.route('/api')
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
        if data1 == "a":
            c1,c2,c3,c4,c5,c6 = 1,0,0,0,0,0
        elif data1 == "b":
            c1,c2,c3,c4,c5,c6 = 0,1,0,0,0,0
        elif data1 == "c":
            c1,c2,c3,c4,c5,c6 = 0,0,1,0,0,0
        elif data1 == "d":
            c1,c2,c3,c4,c5,c6 = 0,0,0,1,0,0
        elif data1 == "e":
            c1,c2,c3,c4,c5,c6 = 0,0,0,0,1,0
        elif data1 == "f":
            c1,c2,c3,c4,c5,c6 = 0,0,0,0,0,1
        else:
            c1,c2,c3,c4,c5,c6 = 1,0,0,0,0,0
        arr1 = [c1,c2,c3,c4,c5,c6]
        data2 = request.form['b']
        if (data2 == "own"):
            c7,c8= 1,0
        elif (data2 == "mortgage"):
            c7,c8= 0,1
        else:
            data2 = 0
        arr2 = [c7,c8]
        data3 = request.form['c']
        if (data3 == "nm"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,1,0,0,0,0,0,0,0,0,0,0,0
        elif (data3 == "ny"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,1,0,0,0,0,0,0,0,0,0,0
        elif (data3 == "ok"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,1,0,0,0,0,0,0,0,0,0
        elif (data3 == "ca"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,0,1,0,0,0,0,0,0,0,0
        elif (data3 == "ut"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,0,0,1,0,0,0,0,0,0,0
        elif (data3 == "ar"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,0,0,0,1,0,0,0,0,0,0
        elif (data3 == "ri"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,0,0,0,0,1,0,0,0,0,0
        elif (data3 == "ga"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,0,0,0,0,0,1,0,0,0,0
        elif (data3 == "wi"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,0,0,0,0,0,0,1,0,0,0
        elif (data3 == "tx"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,0,0,0,0,0,0,0,1,0,0
        elif (data3 == "il"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,0,0,0,0,0,0,0,0,1,0
        elif (data3 == "ks"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,0,0,0,0,0,0,0,0,0,1
        elif (data3 == "wv"):
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,0,0,0,0,0,0,0,0,0,0
        else:
            c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21 = 0,0,0,0,0,0,0,0,0,0,0,0,0
        arr3 = [c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21]
        # data3 means Natu
        data4 = request.form['d']
        if (data4 == "nv"):
            c22,c23 = 1,0
        elif (data4 == "sv"):
            c22,c23 = 0,1
        else:
            c22,c23 = 0,0
        arr4 = [c22,c23]
    # data5 means Mental Peace
        data5 = request.form['e']
        if data5 == "cc":
            c24,c25,c26,c27 = 1,0,0,0,
        elif data5 == "dc":
            c24,c25,c26,c27 = 0,1,0,0,
        elif data5 == "omv":
            c24,c25,c26,c27 = 0,0,1,0,
        elif data5 == "mpc":
            c24,c25,c26,c27 = 0,0,0,1,
        else:
            c24,c25,c26,c27 = 0,0,0,0,
        arr5 = [c24,c25,c26,c27]
    # dat6 means Reaction on lack  of somthing
        data6 = request.form['f']
        if data6 == "w":
            c28= 1
        else:
            c28=0
        arr6 = [c28]
        data7 = request.form['g']
        if data7 == "36":
            c29 = 1
        else:
            c29 = 0
        arr7 = [c29]
        data8 = request.form['h']
        if data8 == "1":
            c30,c31,c32,c33,c34 = 1,0,0,0,0
        elif data8 == "2":
            c30,c31,c32,c33,c34 = 0,1,0,0,0
        elif data8 == "5":
            c30,c31,c32,c33,c34 = 0,0,1,0,0
        elif data8 == "7":
            c30,c31,c32,c33,c34 = 0,0,0,1,0
        elif data8 == "10":
            c30,c31,c32,c33,c34 = 0,0,0,0,1
        else:
            c30,c31,c32,c33,c34 = 0,0,0,0,0
        arr8 = [c30,c31,c32,c33,c34]
        data9 = request.form['i']
        arr.append(data9)
        if data9 == "38":
            c35,c36,c37,c38,c39,c40,c41 = 1,0,0,0,0,0,0
        elif data9 == "39":
            c35,c36,c37,c38,c39,c40,c41 = 0,1,0,0,0,0,0
        elif data9 == "40":
            c35,c36,c37,c38,c39,c40,c41 = 0,0,1,0,0,0,0
        elif data9 == "42":
            c35,c36,c37,c38,c39,c40,c41 = 0,0,0,1,0,0,0
        elif data9 == "49":
            c35,c36,c37,c38,c39,c40,c41 = 0,0,0,0,1,0,0
        elif data9 == "53":
            c35,c36,c37,c38,c39,c40,c41 = 0,0,0,0,0,1,0
        elif data9 == "65":
            c35,c36,c37,c38,c39,c40,c41 = 0,0,0,0,0,0,1
        else:
            c35,c36,c37,c38,c39,c40,c41 = 0,0,0,0,0,0,0
        arr9 = [c35,c36,c37,c38,c39,c40,c41]
        data10 = request.form['j']
        if data10 == "1":
            c42,c43,c44,c45 = 1,0,0,0
        elif data10 == "2":
            c42,c43,c44,c45 = 0,1,0,0
        elif data10 == "3":
            c42,c43,c44,c45 = 0,0,1,0
        elif data10 == "4":
            c42,c43,c44,c45 = 0,0,0,1
        else:
            c42,c43,c44,c45 = 0,0,0,0
        arr10 = [c42,c43,c44,c45]
        data11 = request.form['k']
        if data11 == "1":
            c46,c47,c48,c49,c50 = 1,0,0,0,0
        elif data11 == "2":
            c46,c47,c48,c49,c50 = 0,1,0,0,0
        elif data11 == "3":
            c46,c47,c48,c49,c50 = 0,0,1,0,0
        elif data11 == "4":
            c46,c47,c48,c49,c50 = 0,0,0,1,0
        elif data11 == "5":
            c46,c47,c48,c49,c50 = 0,0,0,0,1
        else:
            c46,c47,c48,c49,c50 = 0,0,0,0,0
        arr11 = [c46,c47,c48,c49,c50]
        data12 = request.form['l']
        if data12 == "0":
            c51,c52,c53 = 1,0,0
        elif data12 == "1":
            c51,c52,c53 = 0,1,0
        elif data12 == "2":
            c51,c52,c53 = 0,0,1
        else:
            c51,c52,c53 = 0,0,0
        arr12 = [c51,c52,c53]
        data13 = request.form['m']
        if data13 == "1":
            c54=1
        else:
            c54=0
        arr13 = [c54]
        data14 = request.form['n']
        if data14 == "1":
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 1,0,0,0,0,0,0,0,0,0,0
        elif data14 == "2":
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 0,1,0,0,0,0,0,0,0,0,0
        elif data14 == "3":
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 0,0,1,0,0,0,0,0,0,0,0
        elif data14 == "4":
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 0,0,0,1,0,0,0,0,0,0,0
        elif data14 == "5":
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 0,0,0,0,1,0,0,0,0,0,0
        elif data14 == "6":
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 0,0,0,0,0,1,0,0,0,0,0
        elif data14 == "7":
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 0,0,0,0,0,0,1,0,0,0,0
        elif data14 == "8":
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 0,0,0,0,0,0,0,1,0,0,0
        elif data14 == "9":
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 0,0,0,0,0,0,0,0,1,0,0
        elif data14 == "10":
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 0,0,0,0,0,0,0,0,0,1,0
        elif data14 == "11":
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 0,0,0,0,0,0,0,0,0,0,1
        else:
            c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65 = 0,0,0,0,0,0,0,0,0,0,0
        arr14 = [c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65]
        data15 = request.form['o']
        if data15 == "1":
            c66,c67,c68,c69,c70,c71,c72,c73 = 1,0,0,0,0,0,0,0
        elif data15 == "2":
            c66,c67,c68,c69,c70,c71,c72,c73 = 0,1,0,0,0,0,0,0
        elif data15 == "3":
            c66,c67,c68,c69,c70,c71,c72,c73 = 0,0,1,0,0,0,0,0
        elif data15 == "4":
            c66,c67,c68,c69,c70,c71,c72,c73 = 0,0,0,1,0,0,0,0
        elif data15 == "5":
            c66,c67,c68,c69,c70,c71,c72,c73 = 0,0,0,0,1,0,0,0
        elif data15 == "6":
            c66,c67,c68,c69,c70,c71,c72,c73 = 0,0,0,0,0,1,0,0
        elif data15 == "7":
            c66,c67,c68,c69,c70,c71,c72,c73 = 0,0,0,0,0,0,1,0
        elif data15 == "8":
            c66,c67,c68,c69,c70,c71,c72,c73 = 0,0,0,0,0,0,0,1
        else:
            c66,c67,c68,c69,c70,c71,c72,c73 = 0,0,0,0,0,0,0,0
        arr15 = [c66,c67,c68,c69,c70,c71,c72,c73]
        data16 = request.form['p']
        if data16 == "0":
            c74,c75,c76,c77,c78 = 1,0,0,0,0
        elif data16 == "1":
            c74,c75,c76,c77,c78 = 0,1,0,0,0
        elif data16 == "2":
            c74,c75,c76,c77,c78 = 0,0,1,0,0
        elif data16 == "3":
            c74,c75,c76,c77,c78 = 0,0,0,1,0
        elif data16 == "4":
            c74,c75,c76,c77,c78 = 0,0,0,0,1
        else:
            c74,c75,c76,c77,c78 = 0,0,0,0,0
        arr16 = [c74,c75,c76,c77,c78]
        data17 = request.form['q']
        if data17 == "0":
            c79,c80,c81,c82,c83,c84 = 1,0,0,0,0,0
        elif data17 == "1":
            c79,c80,c81,c82,c83,c84 = 0,1,0,0,0,0
        elif data17 == "2":
            c79,c80,c81,c82,c83,c84 = 0,0,1,0,0,0
        elif data17 == "3":
            c79,c80,c81,c82,c83,c84 = 0,0,0,1,0,0
        elif data17 == "4":
            c79,c80,c81,c82,c83,c84 = 0,0,0,0,1,0
        elif data17 == "5":
            c79,c80,c81,c82,c83,c84 = 0,0,0,0,0,1
        else:
            c79,c80,c81,c82,c83,c84 = 0,0,0,0,0,0
        arr17 = [c79,c80,c81,c82,c83,c84]
        arr = arr1 + arr2 + arr3 + arr4 + arr5 + arr6 + arr7 + arr8 + arr9 + arr10 + arr11 + arr12 + arr13 + arr14 + arr15 + arr16 + arr17
        arr1 = np.array([arr])
        arr2 = arr1.astype(np.int64)
        pred = model.model.predict(arr2).reshape(1, -1)
        pred_1 = model.model.predict_proba(arr2).reshape(1, -1)
        pred_data = pred_1[0][0]
        pred_data1 = pred_1[0][1]
        print(pred_1)
        return render_template("result.html", data=pred, data1=pred_data, data2=pred_data1)


if __name__ == "__main__":
    app.run(debug=True)
