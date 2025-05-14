import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("wine_cleaned.csv")
X = df.iloc[:, 1:-1].copy().to_numpy()
y = df.iloc[:, -1].copy().to_numpy()
class_names = ['Low', 'Med', 'High']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier(max_depth=40, class_weight='balanced')
clf.fit(X_train, y_train)
print(f"Train Score: {clf.score(X_train, y_train):.3f}")
print(f"Test Score: {clf.score(X_test, y_test):.3f}")

cm = confusion_matrix(y_test, clf.predict(X_test))
disp_cm = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp_cm.plot()
plt.show()

plt.figure(figsize=(35, 35))
plot_tree(clf, filled=True, feature_names=df.columns.to_list(), class_names=class_names, max_depth=3)
plt.show()

# clf = DecisionTreeClassifier(class_weight='balanced')
# parameters = {"max_depth": range(2,64)}
# grid_search = GridSearchCV(clf, param_grid=parameters, cv=5, verbose=1)
# grid_search.fit(X_train, y_train)
# results = pd.DataFrame(grid_search.cv_results_)
# print(results[['param_max_depth', 'mean_test_score', 'rank_test_score']])
# print(f"{grid_search.best_params_}, Score: {grid_search.best_score_}")
# optimal_depth = grid_search.best_params_["max_depth"]






