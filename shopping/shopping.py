import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename, newline='') as file:
        csvfile = csv.reader(file)
        next(csvfile, None)
        evidence = []
        evidence2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        labels = []
        for row in csvfile:
            for i in [0,2,4,11,12,13,14]:
                evidence2[i] = int(row[i])
            for i in [1,3,5,6,7,8,9]:
                evidence2[i] = float(row[i])
            if row[10] == 'Jan':
                evidence2[10] = 0
            elif row[10] == 'Feb':
                evidence2[10] = 1
            elif row[10] == 'Mar':
                evidence2[10] = 2
            elif row[10] == 'Apr':
                evidence2[10] = 3
            elif row[10] == 'May':
                evidence2[10] = 4
            elif row[10] == 'Jun':
                evidence2[10] = 5
            elif row[10] == 'Jul':
                evidence2[10] = 6
            elif row[10] == 'Aug':
                evidence2[10] = 7
            elif row[10] == 'Sep':
                evidence2[10] = 8
            elif row[10] == 'Oct':
                evidence2[10] = 9
            elif row[10] == 'Nov':
                evidence2[10] = 10
            elif row[10] == 'Dec':
                evidence2[10] = 11
            if row[15] == 'Returning_Visitor':
                evidence2[15] = 1
            if row[15] != 'Returning_Visitor':
                evidence2[15] = 0
            if row[16] == 'FALSE':
                evidence2[16] = 0
            elif row[16] == 'TRUE':
                evidence2[16] = 1
            if row[17] == 'TRUE':
                labels.append(1)
            elif row[17] == 'FALSE':
                labels.append(0)
            evidence.append(evidence2[:])
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    neighbors = KNeighborsClassifier(n_neighbors=1)
    neighbors.fit(evidence, labels)
    return neighbors

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    pospos = 0
    pos = 0
    negneg = 0
    neg = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            pos += 1
            if predictions[i] == 1:
                pospos += 1
        if labels[i] == 0:
            neg += 1
            if predictions[i] == 0:
                negneg += 1
    return float(pospos/pos), float(negneg/neg)

if __name__ == "__main__":
    main()
