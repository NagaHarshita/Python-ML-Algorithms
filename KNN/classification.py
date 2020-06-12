import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing

data = pd.read_csv("./car.data", sep=",")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
class1 = le.fit_transform(list(data["class"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))

print(maint)

predict = "class"

x = list(zip(buying, maint, door, persons, safety, lug_boot))
y = list(class1)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)

predict = model.predict(x_test)
names = ["unacc", "acc", "good", "verygood"]


for x in range(len(x_test)):
    print("predicted : ", names[predict[x]], "Data : ", x_test[x], "Actual :", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 7, True)
    print(n)



