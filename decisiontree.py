from knn import readdata, preprocess, missingfeats
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


def dtree(x, y):
    depths = [1, 2, 3, 4, 5, 6, 7, 7]
    folds = KFold(n_splits=8)
    best_model = []
    for (train, valid), d in zip(folds.split(x), depths):
        trainX = x.ix[train]
        trainy = y[train]
        validX = x.ix[valid]
        validy = y[valid]
        dtclf = tree.DecisionTreeClassifier(max_depth=d)
        dtclf = dtclf.fit(trainX, trainy)
        accuracy = dtclf.score(validX, validy)
        y_predict = dtclf.predict(validX)
        fscore = f1_score(validy, y_predict)
        print("Max_depth of tree: {0}, Accuracy: {1}, f1_score: {2}".format(d, accuracy, fscore))
        best_model.append([dtclf, accuracy, fscore, d])
    return best_model

if __name__ == "__main__":
    traindata = readdata("adult.data")
    testdata = readdata("adulttest.test")
    testdata["Salary"] = testdata["Salary"].str.replace(".", "")
    trainX, trainy = preprocess(traindata)
    testX, testy = preprocess(testdata)
    trainX, testX = missingfeats(trainX, testX)
    print(sum(trainy))
    print(sum(testy))
    dtree_model = dtree(trainX, trainy)
    for i in range(len(dtree_model)):
        k = dtree_model[i][0].predict(testX)
        fscore = f1_score(testy, k)
        print("depth= {0}, f1_score= {1}".format(dtree_model[i][3], fscore))
        # j = dtree_model[i][0].score(testX, testy)
        # l = dtree_model[i][0].score(trainX, trainy)
        # print("Train set accuracy: {0}, Error rate: {1}, Max_depth: {2}".format(l, 1 - l, dtree_model[i][3]))
        # print("Test set accuracy: {0}, Error rate: {1}, Max_depth: {2}".format(j, 1 - j, dtree_model[i][3]))
