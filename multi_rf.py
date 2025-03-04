#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_rf.py
~~~~~~~~~~~~~~~~~~~~~
This script uses Random Forest to do the multi-categorize task.

:author: Lei Hu
:email: hul22@mails.tsinghua.edu.cn
:copyright: (c) 2025 by Lei Hu.
:license: MIT License, see LICENSE for more details.
"""

__version__ = "0.1.0"

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import OneClassSVM, SVC
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import auc, accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器

warnings.filterwarnings('ignore')

pic_path = 'temp_pic/'
npy_path = 'temp_npy/'
csv_path = 'temp_csv/'

# read left data
pos = 'left'
file_path_left = csv_path + f'total_data_{pos}.csv'
df_left = pd.read_csv(file_path_left)
labels_left = df_left['label']
df_left = df_left.drop('label', axis=1)

# read right data
pos = 'right'
file_path_right = csv_path + f'total_data_{pos}.csv'
df_right = pd.read_csv(file_path_right)
labels_right = df_right['label']
df_right = df_right.drop('label', axis=1)

# merge data vertically
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

df = pd.concat([df, labels], axis=1)

# split the training set and test set
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.3, random_state=42)

# make the assumption that the label is multi-class now, not only binary

# set the parameters of the Random Forest
clf = RandomForestClassifier(
    n_estimators=280,
    random_state=42,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=1,
    criterion='entropy'
)
# train the model
clf.fit(X_train, y_train)

# predict on the training set
y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Train Accuracy: {train_accuracy:.4f}")

# predict on the test set
y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# get the class list
classes = np.unique(y_train)
n_classes = len(classes)

# predict the class probability
y_test_proba = clf.predict_proba(X_test)

# One-vs-Rest
y_test_binarized = label_binarize(y_test, classes=classes)

# Macro-Averaged AUROC
auroc = roc_auc_score(y_test_binarized, y_test_proba, average="macro", multi_class="ovr")
print(f"Macro-Averaged AUROC: {auroc:.4f}")

# Macro-Averaged AUPRC
auprc = average_precision_score(y_test_binarized, y_test_proba, average="macro")
print(f"Macro-Averaged AUPRC: {auprc:.4f}")

# ROC curve
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_test_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.savefig(pic_path + 'multi_rf_roc_curve.png')

# PR curve
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_test_proba[:, i])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"Class {classes[i]} (AUC = {pr_auc:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.savefig(pic_path + 'multi_rf_pr_curve.png')