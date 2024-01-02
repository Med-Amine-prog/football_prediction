import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import joblib
X = pd.read_csv('full_data.csv')
X = X[X['player_status'] != 'not-called-up']

X = X.dropna()

y = X['player_status']
print(y)


random_seed = 42

# Shuffle both DataFrames using the same random seed
y = y.sample(frac=1, random_state=random_seed)
X = X.sample(frac=1, random_state=random_seed)

X= X.drop('minute_per_game',axis=1)
X = X.drop('balls_recovered_per_game',axis=1)
X = X.drop(X.columns[0],axis=1)
X = X.drop('player_name_sofascore',axis=1)
X = X.drop('player_name_transfer_market',axis=1)
X = X.drop('player_team',axis=1)
X = X.drop('player_status',axis=1)
X = X.drop('player_league',axis=1)
X = X.drop('goals_conceded',axis=1)
X = X.drop('pourcentage passes',axis=1)

print(X.columns)
print(X.info())

label_encoder = LabelEncoder()
# Apply one-hot encoding
X = pd.get_dummies(X, columns=['player_position'], prefix='player_position')
X['national_team'] = label_encoder.fit_transform(X['national_team'])

X['player_market_value'] = X['player_market_value'].fillna(0)
X['league_ranking'] = X['league_ranking'].fillna(X['league_ranking'].mean())
print(X)

# Instantiate LabelEncoder



# Fit and transform the string column
y = label_encoder.fit_transform(y)
print(y)




scaler = MinMaxScaler()

# Fit and transform the DataFrame
# X = scaler.fit_transform(X)
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled_df.to_csv('data_pre.csv')
# X = pd.DataFrame()
X.to_csv('data_pre.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression classifier
logreg_classifier = LogisticRegression(max_iter=8000,C=1.0)

# Train the classifier on the training set
logreg_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')



