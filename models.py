import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

class Model:

    def __init__(self):
        self.name = ''
        path = 'dataset/depressionDataset.csv'
        df = pd.read_csv(path)
        df = df[['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'class']]

        # Handling Missing Data
        df.fillna(df.mode().iloc[0], inplace=True)

        self.split_data(df)
        self.classifier_names = []
        self.accuracies = []

    def split_data(self, df):
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        # Split into 80% train+validation and 20% test
        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=24)
        # Split train+validation into 70% train and 10% validation
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.125, random_state=24)
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def svm_classifier(self):
        self.name = 'SVM Classifier'
        classifier = SVC()
        return classifier.fit(self.x_train, self.y_train)

    def decision_tree_classifier(self):
        self.name = 'Decision Tree Classifier'
        classifier = DecisionTreeClassifier()
        return classifier.fit(self.x_train, self.y_train)

    def random_forest_classifier(self):
        self.name = 'Random Forest Classifier'
        classifier = RandomForestClassifier()
        return classifier.fit(self.x_train, self.y_train)

    def naive_bayes_classifier(self):
        self.name = 'Naive Bayes Classifier'
        classifier = GaussianNB()
        return classifier.fit(self.x_train, self.y_train)

    def knn_classifier(self):
        self.name = 'KNN Classifier'
        classifier = KNeighborsClassifier()
        return classifier.fit(self.x_train, self.y_train)

    def accuracy(self, model):
        predictions = model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        self.classifier_names.append(self.name)
        self.accuracies.append(accuracy * 100)
        print(f"{self.name} has an accuracy of {accuracy * 100:.2f}%")

    def plot_accuracies(self):
        # Sorting the accuracies and names
        sorted_indices = np.argsort(self.accuracies)
        sorted_accuracies = np.array(self.accuracies)[sorted_indices]
        sorted_names = np.array(self.classifier_names)[sorted_indices]

        # Generate a colormap with a unique color for each bar
        colors = cm.get_cmap('tab10', len(sorted_names))

        plt.figure(figsize=(14, 8))  # Increase figure size for better visibility
        bars = plt.bar(sorted_names, sorted_accuracies, color=[colors(i) for i in range(len(sorted_names))])

        plt.xlabel('Classifier')
        plt.ylabel('Accuracy (%)')
        plt.title('Classifier Accuracies')
        plt.ylim(10, 100)

        # Adding labels to the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

        plt.xticks(rotation=30, ha='right', fontsize=12)  # Rotate labels and set fontsize for better readability
        plt.tight_layout()  # Adjust layout to make sure everything fits without overlap
        plt.show()

if __name__ == '__main__':
    model = Model()
    model.accuracy(model.decision_tree_classifier())
    model.accuracy(model.random_forest_classifier())
    model.accuracy(model.naive_bayes_classifier())
    model.accuracy(model.svm_classifier())
    model.accuracy(model.knn_classifier())
    model.plot_accuracies()
