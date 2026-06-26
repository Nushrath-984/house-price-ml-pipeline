import joblib, numpy as np 
m = joblib.load('models/linear_adam.pkl') 
print('w shape:', m.w.shape) 
print('w values:', m.w[:3]) 
print('b:', m.b) 
X = np.ones((1, 11)) 
print('predict:', m.predict(X)) 
