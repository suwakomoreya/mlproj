import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#load dataset
df = pd.read_csv("data/features.csv",index_col=0,parse_dates=True)
df = df.sort_index()
x=df.drop(columns=["next_hour_power"]).to_numpy()
y=df["next_hour_power"].to_numpy()
#chronlog split
split=int(len(df)*0.8)
X_train= x[:split]
X_test=x[split:]

y_train =y[:split]
y_test = y[split:]
#standardized features w training data
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

X_std[X_std==0]=1
X_train = (X_train-X_mean)/X_std
X_test = (X_test-X_mean)/ X_std
# parameters
n_samples,n_features = X_train.shape

weights = np.zeros(n_features)
bias = y_train.mean()

learningrate= 0.03
epochs =3000

for epoch in range(epochs):
    #generate predictins
    predictins = X_train @ weights + bias
    errors = predictins - y_train

    dw = (2/n_samples) *(X_train.T @ errors)
    db = 2*np.mean(errors)

    weights -= learningrate*dw
    bias -= learningrate*db
    if epoch%300 ==0:
        mse = np.mean(errors **2)
        print(f"epoch {epoch}: mse == {mse:.6f}")

predictins = X_test@ weights+bias

#eval
print("\n manual linera regressiona")
print("mae:", mean_absolute_error(y_test,predictins))
print("rmse:", np.sqrt(mean_squared_error(y_test,predictins)))
print("r2:", r2_score(y_test,predictins))

print("\n learned weights")
print(weights)
print("\n learned bias:")
print(bias)