import os
from collections import Counter
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.utils import to_categorical
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
from keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load and preprocess the dataset
data = pd.read_csv('model/crime.csv')

selected_columns = ['incident_t', 'neighborho', 'offense_de']
data = data[selected_columns]

# Handle flexible date parsing
data['incident_t'] = pd.to_datetime(data['incident_t'], errors='coerce')

data['day'] = data['incident_t'].dt.day
data['month'] = data['incident_t'].dt.month
data['year'] = data['incident_t'].dt.year

data.drop(columns=['incident_t'], inplace=True)

# Remove rows with missing values
data.dropna(inplace=True)

# Label encode the 'neighborho' column consistently across both training and test data
label_encoder = LabelEncoder()
data['neighborho'] = label_encoder.fit_transform(data['neighborho'])

# Split the data into training and testing sets
X = data.drop(columns=['offense_de'])
y_offense = data['offense_de']

X_train, X_test, y_train_offense, y_test_offense = train_test_split(
    X, y_offense, test_size=0.2, random_state=42
)



# Train the decision tree model for predicting offense description
offense_model = DecisionTreeClassifier(random_state=42)
offense_model.fit(X_train, y_train_offense)

# Train the k-NN model for predicting offense description
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
knn_model.fit(X_train, y_train_offense)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


# Build the DNN model
input_shape = X_train_scaled.shape[1]
num_classes = len(np.unique(y_train_offense))


# Build the DNN model
def build_dnn(input_shape, num_classes):
    print('building dnn', '*\n'*5)
    input_layer = Input(shape=input_shape, name='input_layer')


    # Domain-specific input branches
    branch_1 = Dense(64, activation='relu')(input_layer)
    branch_2 = Dense(64, activation='relu')(input_layer)
    branch_3 = Dense(64, activation='relu')(input_layer)
    # Add more branches for different domains

    # Combine domain branches
    combined = Concatenate()([branch_1, branch_2, branch_3])

    # Additional hidden layers
    hidden = Dense(128, activation='relu')(combined)
    hidden = Dropout(0.3)(hidden)
    hidden = Dense(64, activation='relu')(hidden)

    # Output layer
    output_layer = Dense(num_classes, activation='softmax', name='output_layer')(hidden)


    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# ========================================================================================================================================

# Check if the saved model exists
saved_model_path = 'model/dnn_model.h5'
label_encoder_path = 'model/label_encoder.joblib'

if os.path.exists(saved_model_path) and os.path.exists(label_encoder_path):
    # Load the label encoder
    label_encoder = joblib.load(label_encoder_path)
    
    # Load the model architecture and weights
    dnn_model = build_dnn(X_train_scaled.shape[1], len(label_encoder.classes_))
    dnn_model.load_weights(saved_model_path)

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    dnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[categorical_accuracy])
    
    print("Loaded existing model.")
else:
    print("No existing model found. Building and training the model.")
    
    # Build the DNN model
    input_shape = X_train_scaled.shape[1]
    num_classes = len(np.unique(y_train_offense))
    dnn_model = build_dnn(input_shape, num_classes)

    # Convert class labels to categorical format
    y_train_categorical = to_categorical(label_encoder.transform(y_train_offense))

    # Scale the test data using the same scaler used for training data
    X_test_scaled = scaler.transform(X_test)

    # Compile and train the model
    optimizer = Adam(learning_rate=0.001)
    dnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[categorical_accuracy])
    dnn_model.fit(X_train_scaled, y_train_categorical, batch_size=64, epochs=20, validation_split=0.2)

    # Save the label encoder and trained model
    joblib.dump(label_encoder, label_encoder_path)
    dnn_model.save_weights(saved_model_path)
# =======================================================================================================================================


# Combine training and test data
combined_data = pd.concat([X_train, X_test])
combined_labels = pd.concat([y_train_offense, y_test_offense])

# Combine training and test data
combined_data = pd.concat([X_train, X_test])
combined_labels = pd.concat([y_train_offense, y_test_offense])

# Label encode the combined 'neighborho' column consistently
label_encoder = LabelEncoder()
combined_labels_encoded = label_encoder.fit_transform(combined_labels)

# Split the combined data back into training and test sets
X_train_combined = combined_data.iloc[:len(X_train)]
X_test_combined = combined_data.iloc[len(X_train):]
y_train_combined_encoded = combined_labels_encoded[:len(X_train)]
y_test_combined_encoded = combined_labels_encoded[len(X_train):]

num_classes = len(np.unique(combined_labels_encoded))

y_train_categorical = to_categorical(y_train_combined_encoded, num_classes=num_classes)
y_test_categorical = to_categorical(y_test_combined_encoded, num_classes=num_classes)

# Scale the combined data
scaler = StandardScaler()
X_train_scaled_combined = scaler.fit_transform(X_train_combined)
X_test_scaled_combined = scaler.transform(X_test_combined)

input_shape_combined = X_train_scaled_combined.shape[1]

# Compile the model for combined data
dnn_model_combined = build_dnn(input_shape_combined, num_classes)

# Convert class labels to categorical format
y_train_categorical_combined = to_categorical(y_train_combined_encoded, num_classes=num_classes)

# Compile the model for combined data
optimizer = Adam(learning_rate=0.001)
dnn_model_combined.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[categorical_accuracy])

# Evaluate the model on test data
X_test_scaled_combined = scaler.transform(X_test_combined)
y_test_categorical_combined = to_categorical(y_test_combined_encoded, num_classes=num_classes)
test_loss_combined, test_accuracy_combined = dnn_model_combined.evaluate(X_test_scaled_combined, y_test_categorical_combined)
print(f"Test Loss (Combined Data): {test_loss_combined:.4f}, Test Accuracy (Combined Data): {test_accuracy_combined:.4f}")

# Print the evaluation results for the combined data
print(f"Test Loss (Combined Data): {test_loss_combined:.4f}, Test Accuracy (Combined Data): {test_accuracy_combined:.4f}")



@app.route('/predict', methods=['POST'])
def predict():
    try:
        df = pd.read_csv('model/crime.csv')
        Data = request.get_json()

        day = int(Data['day'])
        month = int(Data['month'])
        year = int(Data['year'])
        neighborhood = Data['neighborhood']

        # Preprocess the input Data
        new_input = pd.DataFrame({
            'day': [day],
            'month': [month],
            'year': [year],
            'neighborho': [neighborhood]
        })

        # Prepare a list to store the data

        neighb_entries = df[df['neighborho'].str.contains(neighborhood, case=False, na=False)]

        # Prepare a list to store the data
        crime_hist = [ ]
        iteration_count = 0

    # Iterate through the filtered rows and add them to the list
        for index, row in neighb_entries.iterrows():
            crime_hist.append({
                'neighborhood': row['neighborho'],
                'address': row['address'],
                'offense_ca': row['offense_ca'],
                'incident_t': row['incident_t']
            })
            iteration_count += 1

            # Check if the counter has reached 20, and break out of the loop if so
            if iteration_count >= 20:
                break


        new_input_encoded = pd.get_dummies(new_input, columns=['neighborho'])
        new_input_encoded = new_input_encoded.reindex(columns=X_train.columns, fill_value=0)
        print("Encoded Input Data:")
        print(new_input_encoded)

        new_input_scaled = scaler.transform(new_input_encoded)

        print("Scaled Input Data:")
        print(new_input_scaled)

        dnn_predicted_class = np.argmax(dnn_model.predict(new_input_scaled), axis=1)[0]

        print(f"\nDNN Predicted Class: {dnn_predicted_class}")

        # Print KNN and Decision Tree predictions without modification
        knn_predicted_class = knn_model.predict(new_input_encoded)[0]
        decision_tree_predicted_class = offense_model.predict(new_input_encoded)[0]

        result = {
            "dnn_predicted_class": int(dnn_predicted_class),
            "knn_predicted_class": str(knn_predicted_class),
            "decision_tree_predicted_class": str(decision_tree_predicted_class),
            "final_predicted_class": str(knn_predicted_class),
            "city": str(neighborhood),
            "historical_crimes": crime_hist  # Use the list directly
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(debug=True)
