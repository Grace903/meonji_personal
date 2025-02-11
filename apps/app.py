from flask import Flask,render_template,request
import pickle
import numpy as np
# from apps.learning.preprocessing import preprocess_data

app = Flask(__name__, template_folder="templates")

@app.route('/learning')
def learning():
  return render_template('pollutants.html')

@app.route('/test')
def test():
  return render_template('test.html')

# 예측 모델 보내기
# model = pickle.load(open(''))
# @app.route('/predict', methods=['POST'])
# def predictModel():
#   data1 = request.form['a'] -> 입력값
#   data2 = request.form['b']

#   arr = np.arry([[data1,data2]]) 
#   pred = model.predict(arr)
#   return render_template('predict.html', data=pred)

if __name__ == '__main__':
  app.run(debug=True)