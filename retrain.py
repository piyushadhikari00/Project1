import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Downloading dataset...")
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

X = df.drop('medv', axis=1)
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

pickle.dump(model,  open('housepred.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl',    'wb'))

print("✅ housepred.pkl and scaler.pkl created!")
print("✅ Model Score:", round(model.score(scaler.transform(X_test), y_test), 4))