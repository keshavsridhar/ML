from knn import readdata, missingfeats
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score


def preprocess(data):
    y = data.ix[:, -1]
    data = data.ix[:, :-1]
    y = (y == " >50K") * 1
    a = data[data["occupation"] == " ?"].index
    for i in a:
        s = data["occupation"][data["education"] == data.ix[i, "education"]].mode()
        data.ix[i, "occupation"] = s[0]
    w = data[data["workclass"] == " ?"].index
    for i in w:
        s = data["workclass"][data["occupation"] == data.ix[i, "occupation"]].mode()
        data.ix[i, "workclass"] = s[0]
    r = data[data["native-country"] == " ?"].index
    for i in r:
        s = data["native-country"][data["race"] == data.ix[i, "race"]].mode()
        data.ix[i, "native-country"] = s[0]

    # Factorizing the categorical data
    categorical_feats = data.dtypes[data.dtypes == "object"].index
    for i in categorical_feats:
        data[i] = data[i].factorize()[0]
    # Normalizing the data so that no particular column overshadows other column(s)
    numeric_feats = data.dtypes[data.dtypes != "object"].index
    data[numeric_feats] = (data[numeric_feats] - data[numeric_feats].mean()) / data[numeric_feats].std()

    return data, y


def naivebayes(x, y):
    folds = KFold(n_splits=10)
    best_model = []
    for train, valid in folds.split(x):
        trainX = x.ix[train]
        trainy = y[train]
        validX = x.ix[valid]
        validy = y[valid]
        nbclf = GaussianNB()
        nbclf = nbclf.fit(trainX, trainy)
        accuracy = nbclf.score(validX, validy)
        y_pred = nbclf.predict(validX)
        fscore = f1_score(validy, y_pred)
        best_model.append([nbclf, accuracy, fscore])
        print("Accuracy: {0}, f1_score: {1}".format(accuracy, fscore))
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
    nb_model = naivebayes(trainX, trainy)
    for i in range(len(nb_model)):
        k = nb_model[i][0].predict(testX)
        fscore = f1_score(testy, k)
        print("f1_score= {0}".format(fscore))
        # Uncomment the following lines to get accuracies and error rates.
        # j = nb_model[i][0].score(testX, testy)
        # l = nb_model[i][0].score(trainX, trainy)
        # print("Train set accuracy: {0}, Error rate: {1}".format(l, 1 - l))
        # print("Test set accuracy: {0}, Error rate: {1}".format(j, 1 - j))
