# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Φόρτωση του dataset Iris
iris = datasets.load_iris()

# Μετατροπή σε DataFrame για εύκολη επεξεργασία
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target  # Προσθήκη των ετικετών (0, 1, 2)

# Εμφάνιση των πρώτων 5 γραμμών
print(df.head())
