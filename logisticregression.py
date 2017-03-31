from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from knn import readdata, preprocess, missingfeats


def logreg(x, y):
    folds = KFold(n_splits=5)
    tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    # best_model = [None, None, float("-inf"), float("-inf")]
    best_model = []
    for (train, valid), t in zip(folds.split(x), tols):
        trainX = x.ix[train]
        trainy = y[train]
        validX = x.ix[valid]
        validy = y[valid]
        lrclf = LogisticRegression(tol=t)
        lrclf = lrclf.fit(trainX, trainy)
        accuracy = lrclf.score(validX, validy)
        y_pred = lrclf.predict(validX)
        fscore = f1_score(validy, y_pred)
        print("For tolerance level: {0}, Accuracy:{1}, f_1 score:{2}".format(t, accuracy, fscore))
        # if accuracy > best_model[1]:
        #     best_model = [lrclf, accuracy]
        best_model.append([lrclf, t, accuracy, fscore])
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
    logsreg_model = logreg(trainX, trainy)
    for i in range(len(logsreg_model)):
        k = logsreg_model[i][0].predict(testX)
        fscore = f1_score(testy, k)
        print("tol= {0}, f1_score= {1}".format(logsreg_model[i][1], fscore))
        # j = logsreg_model[i][0].score(testX, testy)
        # l = logsreg_model[i][0].score(trainX, trainy)
        # print("Tolerance level: {0}, Test set accuracy: {1}, Test set error: {2}"
        #       .format(logsreg_model[i][1], j, 1 - j))
        # print("Tolerance level: {0}, Train set accuracy: {1}, Train set error: {2}"
        #       .format(logsreg_model[i][1], l, 1 - l))
