import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np


df = pd.read_csv('vgsales.csv')


df = df.dropna(subset=['Year', 'Genre', 'Platform', 'Publisher'])


le_platform = LabelEncoder()
df['Platform'] = le_platform.fit_transform(df['Platform'])

le_genre = LabelEncoder()
df['Genre'] = le_genre.fit_transform(df['Genre'])

le_publisher = LabelEncoder()
df['Publisher'] = le_publisher.fit_transform(df['Publisher'])


X = df[['Year', 'Genre', 'NA_Sales', 'JP_Sales']]
y = df['Platform']


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
rounded_predictions = np.round(predictions).astype(int)


rounded_predictions = np.clip(rounded_predictions, 0, len(le_platform.classes_) - 1)


accuracy = np.mean(rounded_predictions == y_test)
print(f"Accuracy (approx): {accuracy}")



new_game = [2010, 'Shooter', 24.65, 10.05]


encoded_genre = le_genre.transform([new_game[1]])[0]


input_data = [[new_game[0], encoded_genre, new_game[2], new_game[3]]]


pred = model.predict(input_data)
pred_label = int(round(pred[0]))
pred_label = np.clip(pred_label, 0, len(le_platform.classes_) - 1)

predicted_platform = le_platform.inverse_transform([pred_label])
print(f"Predicted Platform (Linear Regression): {predicted_platform[0]}")

