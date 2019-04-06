import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ctx ='C:/Users/ezen/PycharmProjects/Day0330_2/titanic/Data/'
train = pd.read_csv(ctx+'train.csv')
test=pd.read_csv(ctx+'test.csv')
df=pd.DataFrame(train)
print(df.columns)

# PassengerId
#survival	생존여부 Survival	0 = No, 1 = Yes
#pclass	   승선권 클래스Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
#sex	Sex	  성별
#Age	Age   나이 in years
#sibsp	  동반한 형제자매, 배우자 수# of siblings / spouses aboard the Titanic
#parch	  동반한 부모, 자식 수# of parents / children aboard the Titanic
#ticket	  티켓 번호 Ticket number
#fare	  티켓의 요금 Passenger fare
#cabin	  객실번호 Cabin number
#embarked  승선한 항구명 Port of Embarkation
#C = Cherbourg 쉐부로, Q = Queenstown퀸스타운, S = Southampton 사우스햄톤


f, ax = plt.subplots(1,2,figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],
                                      autopct="%1.1f%%",ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')

sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Survived')
# plt.show() 생존률 38.4% / 사망률 61.6%

"""
데이터는 훈련이다(train.csv), 목적데이터(test.csv
)둣가지로 데송됩니다
목적데이터에는 위 항목에서는 Survived 정보가 빠져있습니다.
그것은 답이기 때문입니다.
"""

# *******
# 성별
# *******


#f, ax = plt.subplots(1,2,figsize=(18,8))

#train['Survived'][train["sex"]=='male'].value_counts().plot.pie(explode=[0,0.1],
#                                      autopct="%1.1f%%",ax=ax[0],shadow=True)
#train['Survived'][train["sex"]=='female'].value_counts().plot.pie(explode=[0,0.1],
#                                      autopct="%1.1f%%",ax=ax[1],shadow=True)
#ax[0].set_title('Survived(male)')
#ax[1].set_title('Survived(female)')


# plt.show()  남성생존율 81.1 사망율 18.9
                         # 74.2        25.8

# *******
# 승선권 Pclass
# *******

"""
df_1 =[train['Sex'],train['Survived']]
df_2 = train['Pclass']
pd.crosstab(df_1,df_2,margins=True)
print(df.head())
"""
"""
f, ax = plt.subplots(2,2,figsize=(20,15))
sns.countplot('Pclass',data=train,ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Pclass',hue='Sex',data=train,ax=ax[0,1])
ax[0,1].set_title('Male  / Female Embarked')
sns.countplot('Embarked', hue='Survived', data=train,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Pclass',data=train,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')

plt.show()



train['Survived'].value_counts().plot.pie(explode=[0,0.1],
                                      autopct="%1.1f%%",ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')

sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Survived')
"""

# 결측치 제거

# train.info()
"""
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
"""

# print(train.isnull().sum())
"""
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
"""

def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead= train[train['Survived'] == 0][feature].value_counts()
    df=pd.DataFrame([survived, dead])
    df.index=['survived','dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
    plt.show()

#bar_chart('Sex')
#bar_chart('Pclass')
#bar_chart('SibSp')
#bar_chart('Parch')
#bar_chart('Embarked')

#Cabin, Ticket값 삭제
train=train.drop(['Cabin'],axis=1)
test=test.drop(['Cabin'],axis=1)
train=train.drop(['Ticket'],axis=1)
test=test.drop(['Ticket'],axis=1)
print(train.head())
print(test.head())

# Embarked값 가공

s_city=train[train['Embarked']=='S'].shape[0]  #스칼라
c_city=train[train['Embarked']=='C'].shape[0]
q_city=train[train['Embarked']=='Q'].shape[0]

#print("S=",s_city)
#print("C=",c_city)
#print("Q=",q_city)

print("S={}.C={},Q={}".format(s_city,c_city,q_city))


train=train.fillna({"Embarked":"S"})
city_mapping={"S":1,"C":2,"Q":3}
train['Embarked']=train["Embarked"].map(city_mapping)
test['Embarked']=train["Embarked"].map(city_mapping)

#print(train.head())
#print(test.head())

# Name 값 가공하기

combine=[train,test]
for dataset in combine:
    dataset ['Title']= dataset.Name.str.extract('[A-Za-z]\.',expand=False)

pd.crosstab(train['Title'],train['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mne', 'Mrs')
#print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())

train = train.drop(['Name','PassengerId'],axis=1)
test = train.drop(['Name','PassengerId'],axis=1)

combine =[train,test]
print(train.head())


sex_mapping={"male":0,"Female":1}
for dataset in combine:
    dataset['Sex']=dataset['Sex'].map(sex_mapping)

title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5,'Rare':6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) # fillna
train.head()

# Age 값 가공하기

train['Age']=train['Age'].fillna(-0.5)
test['Age']=test['Age'].fillna(-0.5)
bins=[-1,0,5,12,18,24,35,60,np.inf]
labels =['Unknown','Baby','Child','Teenager','Student','Young Adult','Audlt','Senior']
train['AgeGroup']=pd.cut(train['Age'],bins,labels=labels)
test['AgeGroup']=pd.cut(test['Age'],bins,labels=labels)

print(train.head())

age_mapping={'Baby' : 1, 'Child':2, 'Teenager':3, 'Student': 4, 'Young Adult':5,'Adult':6, 'Senior':7}
"""
for x in range(len(train['AgeGroup'])):
    if train['AgeGroup'][x]=='Unknown':
        train['AgeGroup'][x]=age_title_mapping[train['Title'][x]]

for x in range(len(test['AgeGroup'])):
    if test['AgeGroup'][x] == "Unknown":
        test['AgeGroup'][x]=age_title_mapping[test['Title'][x]]

"""

age_title_mapping = {0:"Unknon",1: "Young Adult", 2:"Student", 3:"Adult", 4:"Baby", 5:"Adult", 6:"Adult"}
for x in range(len(train['AgeGroup'])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train['Title'][x]]
for x in range(len(test['AgeGroup'])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test['Title'][x]]


age_mapping={'Baby' : 1, 'Child':2, 'Teenager':3, 'Student': 4, 'Young Adult':5,'Adult':6, 'Senior':7}
train['AgeGroup']=train['AgeGeoup'].map(age_mapping)
test['AgeGroup']=test['AgeGeoup'].map(age_mapping)
train=train.drop(['Age'],axis=1)
test=test.drop(['Age'],axis=1)

# Fare 처리

train ['FareBand']=pd.qcut(train['Fare'],4,labels={1,2,3,4})
test ['FareBand']=pd.qcut(train['Fare'],4,labels={1,2,3,4})

train=train.drop(['Fare'],axis=1)
test=test.drop(['Fare'],axis=1)

"""

for x in range(len(train['AgeGroup'])):
    if train['AgeGroup'][x]=='Unknown':
        train['AgeGroup'][x]=age_title_mapping[train['Title'][x]]

for x in range(len(test['AgeGroup'])):
    if test['AgeGroup'][x] == "Unknown":
        test['AgeGroup'][x]=age_title_mapping[test['Title'][x]]

"""

#**************************
# 데이터 모델링
#**************************

train.data=train.drop('Sirvived',axis=1)
target=train['Survived']
print(train.data.shape)
print(target.shape)

print(train.info_)





