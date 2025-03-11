# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Φόρτωση του dataset Iris
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 2️⃣ Προσθήκη ονομάτων στα είδη λουλουδιών
species = ['Setosa', 'Versicolor', 'Virginica']
df['species'] = df['target'].apply(lambda x: species[x])

# 3️⃣ Οπτικοποίηση των δεδομένων (Scatter Plot)
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['petal length (cm)'], y=df['petal width (cm)'], hue=df['species'], palette='viridis')
plt.xlabel('Μήκος Πετάλου (cm)')
plt.ylabel('Πλάτος Πετάλου (cm)')
plt.title('Κατανομή των λουλουδιών ανάλογα με το μέγεθος των πετάλων')
plt.legend(title="Είδος")
plt.show()

# 4️⃣ Διαχωρισμός Δεδομένων σε Features & Target
X = df.iloc[:, :-2]  # Επιλέγουμε όλες τις στήλες εκτός από τις target/species
y = df['target']  # Τα labels

# 5️⃣ Διαχωρισμός σε Training & Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Κανονικοποίηση των Δεδομένων
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7️⃣ Εκπαίδευση Μοντέλου SVM
model = SVC(kernel='linear')  # Χρησιμοποιούμε SVM με γραμμικό πυρήνα
model.fit(X_train, y_train)

# 8️⃣ Πρόβλεψη στο Test Set
y_pred = model.predict(X_test)

# 9️⃣ Αξιολόγηση Μοντέλου
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Ακρίβεια Μοντέλου: {accuracy:.2f}")

# Αναλυτική αναφορά απόδοσης
print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred, target_names=species))

# 1️⃣0️⃣ Οπτικοποίηση Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d', xticklabels=species, yticklabels=species)
plt.xlabel('Προβλεπόμενη Κλάση')
plt.ylabel('Πραγματική Κλάση')
plt.title('Confusion Matrix')
plt.show()
