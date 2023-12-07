import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import os 
import sys
import yaml

class KeystrokeDynamicsEvaluator:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.model_trained = False
        self.user = None
        self.output_dir = "output"

    def user_accept(self, user):
        self.user = user

    def model_training_status(self):
    	yaml_file_path = os.path.join(self.output_dir, "model_status.yaml")
    	data = {"model":"trained"}
    	with open(yaml_file_path, "w") as yaml_file:
    		yaml.dump(data, yaml_file)

    def load_data(self, training_data_path, testing_data_path):
        if not os.path.isfile("output/model_status.yaml"):
            df = pd.read_csv(training_data_path)
            df_to_predict = pd.read_csv(testing_data_path)

            numerical_values = df_to_predict.values.tolist()
            actual_values = numerical_values[0]

            X = df.drop(['subject', 'sessionIndex', 'rep'], axis=1)
            y = df['subject']

            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, random_state=42)

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.model = Sequential()
            self.model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))

            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            self.model.fit(X_scaled, y_encoded, epochs=10, batch_size=32)

            X_test_scaled = self.scaler.transform(X_test)
            accuracy = self.model.evaluate(X_test_scaled, y_test)[1]
            print(f'\nModel Accuracy: {accuracy}')

            self.model_training_status()

        # X_to_predict_scaled = self.scaler.transform(np.array(actual_values, ndmin=2))
        # prediction = self.model.predict(X_to_predict_scaled)

        # Convert the predicted class to the original label using the label encoder
        predicted_label = self.label_encoder.classes_[np.argmax(prediction)]

        # Print the result
        if self.user == predicted_label:
            print(f"\n[-] Welcome user: {predicted_label}")
        else:
            print(f"\n[!] Intrusion detected..")
            sys.exit()

# Do not include the example usage here, it was removed as requested.
