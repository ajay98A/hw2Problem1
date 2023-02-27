import numpy as np
from numpy import mean,std
import pandas as pd
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from IPython.display import display
import sklearn.tree as tree
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

#Train data
print("Train data")
print(trainData.head())
print(trainData.isna().sum())
print(trainData.index)
print(trainData.columns)
print(trainData.dtypes)
print(trainData.describe())

#Test data
print("\nTest Data")
print(testData.head())
print(testData.isna().sum())
print(testData.index)
print(testData.columns)
print(testData.dtypes)
print(testData.describe())

#Data Preprocess
#Combine test and train
combined = trainData.append(testData)
#Children with title Master
display(combined[(combined.Age.isnull()) & (combined.Name.str.contains('Master'))])
#printing mean age of children
print(trainData[trainData.Name.str.contains('Master')]['Age'].mean()) #5
########children traavlling without parents     
display((combined[(combined.Age.isnull()) & (combined.Name.str.contains('Master')) & (combined.Parch==0)])) 
#1 child lets assume his age to be 15
testData.loc[testData.PassengerId == 1231, 'Age'] = 15
trainData['Title'], testData['Title'] = [dataset.Name.str.extract (' ([A-Za-z]+)\.', expand=False) for dataset in [trainData, testData]]
print(trainData.groupby(['Title', 'Pclass'])['Age'].agg(['mean', 'count']))

#Consolidating titles
titleD = {"Capt": "Officer","Col": "Officer","Major": "Officer","Jonkheer": "Royalty", \
             "Don": "Royalty", "Sir" : "Royalty","Dr": "Royalty","Rev": "Royalty", \
             "Countess":"Royalty", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs","Mr" : "Mr", \
             "Mrs" : "Mrs","Miss" : "Miss","Master" : "Master","Lady" : "Royalty"}

trainData['Title'], testData['Title'] = [df.Title.map(titleD) for df in [trainData, testData]]
print(trainData.groupby(['Title', 'Pclass'])['Age'].agg(['mean', 'count']))

#test and train view
display(trainData[trainData.Title.isnull()])
display(testData[testData.Title.isnull()])

#Donna title update
testData.at[414,'Title'] = 'Royalty'
#Test data
display(testData[testData.Title.isnull()])

#Filling fare
display(trainData[trainData.Fare.isnull()])
display(testData[testData.Fare.isnull()])

trainData["Fare"].fillna(trainData["Fare"].mean(),inplace=True)
testData["Fare"].fillna(testData["Fare"].mean(),inplace=True)

print("Testing")
display(testData[testData.Fare.isnull()])

print(trainData.groupby(['Pclass','Sex','Title'])['Age'].agg({'mean', 'median', 'count'}))

combined = trainData.append(testData)
print(combined.columns)

for df in [trainData, testData, combined]:
    df.loc[(df.Title=='Miss') & (df.Parch!=0), 'Title']="FemaleChild"

display(combined[(combined.Age.isnull()) & (combined.Title=='FemaleChild')])

grp = trainData.groupby(['Pclass','Sex','Title'])['Age'].mean()
print(grp)

print(type(grp)) #type = series object in pandas so convert it to df

grp = trainData.groupby(['Pclass','Sex','Title'])['Age'].mean().reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

print('\n', 'Lookup', grp[(grp.Pclass==2) & (grp.Sex=='male') & (grp.Title=='Master')]['Age'])
#Get only first value

print('\n', 'First value to get age', grp[(grp.Pclass==2) & (grp.Sex=='male') & (grp.Title=='Master')]['Age'].values[0])

#Filling missing values of ages
def fill_age(x):
    return grp[(grp.Pclass==x.Pclass)&(grp.Sex==x.Sex)&(grp.Title==x.Title)]['Age'].values[0]
trainData['Age'], testData['Age'] = [df.apply(lambda x: fill_age(x) if np.isnan(x['Age']) else x['Age'], axis=1) for df in [trainData, testData]]

combined=trainData.append(testData)

#We have some Null values in embarked. So filling them
print("Testing train data embarked column\n")
display(trainData[trainData.Embarked.isnull()])
print("Testing test data embarked column")
display(testData[testData.Embarked.isnull()])


freq_port = trainData.Embarked.dropna().mode()[0]
print(freq_port)

# for ds in combined:
#     dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

for df in [trainData, testData,combined]:
    df['Embarked'].fillna(freq_port,inplace=True)

print("Testing train data embarked column\n")
display(trainData[trainData.Embarked.isnull()])
print("Testing test data embarked column")
display(testData[testData.Embarked.isnull()])

trainData['Embarked'] = trainData['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
testData['Embarked'] = testData['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

trainData['Sex'] = trainData['Sex'].map({'female': 1, 'male': 0}).astype(int)
testData['Sex'] = testData['Sex'].map({'female': 1, 'male': 0}).astype(int)


#Dropping unnecessary values

for df in [trainData, testData, combined]:
    df['familySize'] = df['SibSp'] + df['Parch'] + 1

print(trainData.head())

for df in [trainData, testData, combined]:
    df['isAlone'] = 0
    df.loc[df['familySize'] == 1, 'isAlone'] = 1


trainData = trainData.drop(['Ticket', 'Cabin'], axis=1)
testData = testData.drop(['Ticket', 'Cabin'], axis=1)

print(trainData.groupby(['Title']).agg(['count']))

title_mapping = {"Mr": 1, "Mrs": 2, "Miss": 3, "FemaleChild": 4, "Master": 5, "Royalty": 6, "Officer": 7}
trainData['Title'] = trainData['Title'].map(title_mapping).astype(int)
testData['Title'] = testData['Title'].map(title_mapping).astype(int)

trainData['AgeBand'] = pd.cut(trainData['Age'], 5)
print(trainData[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index = False).mean())

for dataset in [trainData, testData, combined]:
    dataset.loc[ dataset['Age'] <= 16.33, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16.33) & (dataset['Age'] <= 32.25), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32.25) & (dataset['Age'] <= 48.16), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48.16) & (dataset['Age'] <= 64.08), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)
trainData = trainData.drop(['AgeBand'], axis=1)
print(testData.head(10))


trainData['FareBand'] = pd.qcut(trainData['Fare'], 5)
print(trainData[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

for dataset in [trainData, testData, combined]:
    dataset.loc[ dataset['Fare'] <= 7.85, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.85) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.67), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 21.67) & (dataset['Fare'] <= 39.68), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 39.68, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)
trainData = trainData.drop(['FareBand'], axis=1)
print(trainData.head(10))
print(testData.head(10))

trainData = trainData.drop(['Name', 'PassengerId', 'Parch', 'SibSp', 'familySize'], axis=1)
testData = testData.drop(['Name','Parch', 'SibSp', 'familySize'], axis=1)

print(trainData.head(10))
print(testData.head(10))


#########################################

X_train = trainData.drop("Survived", axis=1)
Y_train = trainData["Survived"]
X_test  = testData.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)
plt.figure(figsize=(25,20))
tree.plot_tree(decision_tree.fit(X_train,Y_train), feature_names=trainData.columns, max_depth = 3, filled = True)
plt.show()

#Five fold cross validation of decision tree

cross_val = KFold(n_splits = 6, random_state=1, shuffle=True)
score = cross_val_score(decision_tree,X_train,Y_train, scoring='accuracy', cv=cross_val, n_jobs=-1)

print('average classification accuracy of decision tree: %.2f' %(mean(score)))

#Random forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)


#Five fold cross validation of decision tree
cross_val = KFold(n_splits = 6, random_state=1, shuffle=True)
score = cross_val_score(random_forest,X_train,Y_train, scoring='accuracy', cv=cross_val, n_jobs=-1)
print('average classification accuracy of random forest: %.2f' %(mean(score)))


submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)
