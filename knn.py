from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import pandas as pd
import sklearn.neighbors as n


def readdata(filename):
    td = pd.read_csv(filename, header=None, delimiter=",", names=["age", "workclass", "fnlwgt", "education",
                                                                  "education-num", "marital-status", "occupation",
                                                                  "relationship", "race", "sex", "capital-gain",
                                                                  "capital-loss", "hours-per-week", "native-country",
                                                                  "Salary"])
    # td = td.drop(td[td["native-country"] == " ?"].index)
    # td = td.drop(td[td["occupation"] == " ?"].index)
    # td = td.drop(td[td["workclass"] == " ?"].index)
    return td


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
    categorical_feats = data.dtypes[data.dtypes == "object"].index
    for i in categorical_feats:
        newdata = pd.get_dummies(data[i], prefix="*"+i)
        data = data.drop(i, axis=1)
        data = data.join(newdata)
    # Normalizing the data so that no particular column overshadows other column(s)
    numeric_feats = data.dtypes[data.dtypes != "object"].index
    # print(data[numeric_feats].head(3))
    data[numeric_feats] = (data[numeric_feats] - data[numeric_feats].mean()) / data[numeric_feats].std()
    # print(data[numeric_feats].head(3))
    return data, y


def knn(X, y):
    # kf = KFold(n_splits=10)
    # ks = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    kf = KFold(n_splits=5)
    ks = [1, 2, 3, 4, 5]
    # best_model = [None, None, float("-inf")]
    best_model = []
    for (train, valid), k in zip(kf.split(X), ks):
        trainX = X.ix[train]
        trainy = y.ix[train]
        validX = X.ix[valid]
        validy = y.ix[valid]
        kclf = n.KNeighborsClassifier(n_neighbors=k)
        kclf = kclf.fit(trainX, trainy)
        accuracy = kclf.score(validX, validy)
        ypred = kclf.predict(validX)
        fscore = f1_score(validy, ypred)
        # print("accuracy for", k, ":", accuracy)
        # print("f1-score for", k, ":", fscore)
        best_model.append([kclf, k, accuracy, fscore])
        # if accuracy > best_model[2]:
        #     # best_model.append([kclf, k, accuracy])
        #     best_model = [kclf, k, accuracy]
        # if fscore > best_model[2]:
        #     # best_model.append([kclf, k, accuracy])
        #     best_model = [kclf, k, fscore]
    for i in range(len(ks)):
        # print("Nearest-neighbour(", i + 1, ") Validation set accuracy: ", best_model[i][2],
        #       " Validation set fscore:", best_model[i][3])
        print("Nearest-neighbor: {0}, Validation set accuracy: {1}, Validation set f1_score: {2}"
              .format(i+1, best_model[i][2], best_model[i][3]))
    return best_model


def missingfeats(train, test):
    a = list(train)
    b = list(test)
    # Adding a dummy feature vector in case either train or test has some missing features.
    # Also aligning the features so that they get matched properly
    for i in a:
        if i not in b:
            x = train[i]
            train.drop([i], axis=1, inplace=True)
            test[i] = 0
            train = train.join(x)
    for i in b:
        if i not in a:
            y = test[i]
            test.drop([i], axis=1, inplace=True)
            train[i] = 0
            test = test.join(y)
    return train, test

if __name__ == "__main__":
    traindata = readdata("adult.data")
    testdata = readdata("adulttest.test")
    testdata["Salary"] = testdata["Salary"].str.replace(".", "")
    trainX, trainy = preprocess(traindata)
    testX, testy = preprocess(testdata)
    trainX, testX = missingfeats(trainX, testX)
    knn_model = knn(trainX, trainy)
    for i in range(len(knn_model)):
        k = knn_model[i][0].predict(testX)
        fscore = f1_score(testy, k)
        print("k= {0}, f1_score= {1}".format(i + 1, fscore))
        # Uncomment following lines to get accuracy and error rate.
        j = knn_model[i][0].score(testX, testy)
        print("Nearest-neighbor: {0}, Test set accuracy: {1}, Test set error rate: {2}"
              .format(i + 1, j, 1 - j))
