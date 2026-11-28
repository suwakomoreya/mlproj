import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load dataset
df = pd.read_csv("data/features.csv",index_col="datetime",parse_dates=True)
#chronological ordering
df = df.sort_index()

#separate features and targe
X = df.drop(columns=["next_hour_power"])
y= df["next_hour_power"]
#chronological train/test split
split_index = int(len(df)*0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
print("training observations:", len(X_train))
print("testing observations:", len(X_test))
#eval function
def evaluate_model(name,prediction):
    mae = mean_absolute_error(y_test,prediction)
    rmse = np.sqrt(mean_squared_error(y_test,prediction))

    r2 = r2_score(y_test,prediction)
    print(f"\n{name}")
    print("---")
    print(f"MAE: {mae:.4f} kw")
    print(f"rmse: {rmse:.4f} kw")
    print(f"r2: {r2:.4f} kw")

# persistence baseline
baseline_predictions = X_test["current_power"]
evaluate_model("Persistence baseline", baseline_predictions)

# regression
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
linear_predictions = linear_model.predict(X_test)
evaluate_model("Linear regression", linear_predictions)

# random forest
forest_model = RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=1)
forest_model.fit(X_train,y_train)
forest_predictions = forest_model.predict(X_test)
evaluate_model("Random forest", forest_predictions)

# inspect regression weights
print("\nregression coefficients")
for feature, coefficient in zip(X.columns,linear_model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print("intercept:", linear_model.intercept_)

# random forest importnace
print("\nforest feature importnace")
importance = pd.Series(forest_model.feature_importances_,index=X.columns).sort_values(ascending=False)
print(importance)