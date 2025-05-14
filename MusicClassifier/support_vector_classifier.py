import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("music_cleaned.csv")

X = df.iloc[:, 1:-1].to_numpy()
y = df['genre'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = SVC(C=8, kernel = "rbf")
clf.fit(X_train, y_train)
print(f"Train Score: {clf.score(X_train, y_train):.3f}")
print(f"Test Score: {clf.score(X_test, y_test):.3f}")

cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()