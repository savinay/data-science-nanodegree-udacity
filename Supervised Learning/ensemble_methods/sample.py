from sklearn.tree import DecisionTreeClassifier
model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)

#x_train, y_train, x_test will be given

model.fit(x_train, y_train)
model.predict(x_test)