import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv

# Path to the folder containing .npy files
folder_path = "output_features"
label_folder = "label"  # Folder to save the labels

# Step 1: Summarize features for each video
video_summaries = []
video_names = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".npy"):
        file_path = os.path.join(folder_path, file_name)

        try:
            features = np.load(file_path)

            if features.size == 0:
                print(f"Warning: {file_name} is empty, skipping.")
                continue

            # Calculate statistical summaries
            mean_features = np.mean(features, axis=0)
            std_features = np.std(features, axis=0)

            # Combine summaries into a single feature vector
            summarized_features = np.concatenate([mean_features, std_features])
            video_summaries.append(summarized_features)
            video_names.append(file_name)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Convert to numpy array for clustering
if len(video_summaries) > 0:
    video_summaries = np.array(video_summaries)

    # Step 2: Standardize features
    scaler = StandardScaler()
    summaries_scaled = scaler.fit_transform(video_summaries)

    # Step 3: Determine optimal number of clusters using the Elbow Method
    wcss = []  # List to store within-cluster sum of squares
    max_clusters = 15  # Adjust based on your data
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(summaries_scaled)
        wcss.append(kmeans.inertia_)

    # Plot WCSS to find the elbow point
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()

    # Step 4: Apply K-Means clustering with the chosen number of clusters
    optimal_clusters = 4  # Replace with your chosen value based on the elbow plot
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(summaries_scaled)

    # Step 5: Visualize the clusters using PCA
    pca = PCA(n_components=2)
    summaries_2d = pca.fit_transform(summaries_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(summaries_2d[:, 0], summaries_2d[:, 1], c=clusters, cmap='viridis', s=50)
    plt.title(f"K-Means Clustering (n_clusters={optimal_clusters})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

    # Step 6: Save the labels to a CSV file
    os.makedirs(label_folder, exist_ok=True)  # Create the label folder if it doesn't exist
    label_file_path = os.path.join(label_folder, "video_labels.csv")

    with open(label_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Video Name', 'Cluster'])  # Write header

        for video_name, cluster_id in zip(video_names, clusters):
            # Extract the .mp4 file name
            video_name_mp4 = video_name.replace(".npy", ".mp4")
            writer.writerow([video_name_mp4, cluster_id])

    print(f"Labels saved to {label_file_path}")

    # Print cluster assignments
    for video_name, cluster_id in zip(video_names, clusters):
        print(f"Video: {video_name.replace('.npy', '.mp4')} -> Cluster: {cluster_id}")

else:
    print("No valid video summaries found.")
