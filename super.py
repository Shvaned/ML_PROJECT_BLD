import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Step 1: Load features and labels
folder_path = "output_features"
label_file_path = "label/video_labels.csv"  # Path to the CSV containing video labels

# Load video labels
labels_df = pd.read_csv(label_file_path)

# Initialize lists for features and labels
video_features = []
video_labels = []

# Load features from .npy files and match with labels
for file_name in os.listdir(folder_path):
    if file_name.endswith(".npy"):
        file_path = os.path.join(folder_path, file_name)

        try:
            features = np.load(file_path)
            video_name = file_name.replace(".npy", ".mp4")

            # Match with label
            label = labels_df[labels_df['Video Name'] == video_name]['Cluster'].values[0]

            # Add features and label
            if features.size > 0:
                # Flatten features for each video and add to the list
                summarized_features = np.mean(features, axis=0)  # Summarized feature (mean across frames)
                video_features.append(summarized_features)
                video_labels.append(label)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Step 2: Standardize the features
scaler = StandardScaler()
video_features_scaled = scaler.fit_transform(video_features)

# Step 3: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(video_features_scaled, video_labels, test_size=0.2, random_state=42)

# Step 4: Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = clf.predict(X_test)

# Step 6: Evaluate the model
# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(y_train))
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

# Step 7: Feature Importance (Visualization)
feature_importances = clf.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.xticks(range(len(sorted_idx)), sorted_idx, rotation=90)
plt.title("Feature Importance in Random Forest")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

# Step 8: Save the trained model (optional)
import joblib
joblib.dump(clf, 'video_cluster_model.pkl')
# Step 2: Standardize the features
scaler = StandardScaler()
video_features_scaled = scaler.fit_transform(video_features)
# Save the fitted scaler for later use during inference
joblib.dump(scaler, 'scaler.pkl')

