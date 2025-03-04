#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
feature_analysis.py
~~~~~~~~~~~~~~~~~~~~~
This script demonstrates how to use hierarchical clustering to find representative samples for each cluster,
and then use SVM to classify the samples.

:author: Lei Hu, Peixuan Song
:email: spx22@mails.tsinghua.edu.cn; hul22@mails.tsinghua.edu.cn
:copyright: (c) 2025 by Lei Hu, Peixuan Song.
:license: MIT License, see LICENSE for more details.
"""

__version__ = "0.1.0"

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split 

warnings.filterwarnings('ignore')

pic_path = 'temp_pic/'
npy_path = 'temp_npy/'
csv_path = 'temp_csv/'

# set the threshold for clustering
threshold = 0.1

# read the left data
pos = 'left'
file_path_left = csv_path + f'total_data_{pos}.csv'
df_left = pd.read_csv(file_path_left)
labels_left = df_left['label']
df_left = df_left.drop('label', axis=1)

# read the right data
pos = 'right'
file_path_right = csv_path + f'total_data_{pos}.csv'
df_right = pd.read_csv(file_path_right)
labels_right = df_right['label']
df_right = df_right.drop('label', axis=1)

# merge the left and right data
df_left.columns = [col + '_left' for col in df_left.columns]
df_right.columns = [col + '_right' for col in df_right.columns]
df = pd.concat([df_left, df_right], axis=1)
if (labels_left != labels_right).any():
    raise  ValueError("Labels are different.")
labels = labels_left

keys = df.columns.tolist()

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
X = df.values.T


# calculate the distance matrix
distance_matrix = pdist(X, metric='correlation')

# calculate the maximum and minimum distance
Z = linkage(distance_matrix, method='single')
print("Maximum distance:", np.max(Z[:, 2]))
print("Minimum distance:", np.min(Z[:, 2]))


# draw the dendrogram
plt.figure(figsize=(10, 8))
dendrogram(Z)
plt.title("Ward's Hierarchical Clustering Dendrogram")
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.savefig(pic_path+'dendrogram.png')

# find the representative samples for each cluster
clusters = fcluster(Z, t=threshold, criterion='distance')
idx = np.arange(clusters.shape[0])

representative_samples = []
unique_clusters = np.unique(clusters)
for cluster in unique_clusters:
    cluster_samples = X[clusters == cluster]
    idxx = idx[clusters == cluster]
    min_avg_distance = np.inf
    representative_sample = None
    for i, sample in enumerate(cluster_samples):
        distances_to_others = pdist(np.vstack((sample, X)), metric='correlation')
        avg_distance = np.mean(distances_to_others[distances_to_others > 0])
        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            representative_sample = idxx[i]
    representative_samples.append(representative_sample)

# output the representative samples
features_final = []
for ii in representative_samples:
    features_final.append(keys[ii])
    print(f"keys[{ii}] = {keys[ii]}")
print(f"Representative {len(representative_samples)} samples for each cluster (above)")


# select the final features
df_final = df[features_final]
df_final = pd.concat([df_final, labels], axis=1)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_final.drop('label', axis=1), df_final['label'], test_size=0.3, random_state=42)

# train the SVM model
clf = SVC(kernel='linear', C=1.0, gamma='scale', probability=True)
clf.fit(X_train, y_train)


# evaluate the model on the training set
y_pred = clf.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy on train: {accuracy:.4f}")

# evaluate the model on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test: {accuracy:.4f}")

# calculate the AUROC and AUPRC
y_pred_prob = clf.predict_proba(X_test)[:, 1]  # find the probability of the positive class
auroc = roc_auc_score(y_test, y_pred_prob)
print(f"AUROC: {auroc:.4f}")
auprc = average_precision_score(y_test, y_pred_prob)
print(f"AUPRC: {auprc:.4f}")

# draw the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUROC = {auroc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig(pic_path + 'roc_curve.png')

# draw the PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', label=f'Precision-Recall curve (AUPRC = {auprc:.4f})')
plt.title("Precision-Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig(pic_path + 'pr_curve.png')


