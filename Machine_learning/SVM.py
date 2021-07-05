from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
training_points = [[1,3], [3, 3], [4, 0], [0, 0], [1, 2], [2, 0]]
labels = [1, 1, 1, 0, 0, 0]
classifier.fit(training_points, labels)

print(classifier.predict([[3, 0]]))