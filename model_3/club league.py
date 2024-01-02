import pandas as pd

X = pd.read_csv('caf_nation.csv')
y = X['result']
X = X.drop('result',axis=1)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

X['Stage'] = label_encoder.fit_transform(X['Stage'])
y = label_encoder.fit_transform(y)

pd.set_option('display.max_columns', None)

X = X.drop(X.columns[0],axis=1)

print(X.head())

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
scaler = StandardScaler()

# Fit and transform the DataFrame
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the classifier on the training set
gb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Gradient Boosting Accuracy: {accuracy}')







