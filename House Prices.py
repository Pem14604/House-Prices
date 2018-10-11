


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_train = pd.read_csv('train2.csv')
test = pd.read_csv('test.csv')
test_original = pd.read_csv('test.csv')
sub = pd.read_csv('sub.csv')

df_train=df_train.drop('Id', axis=1)
test=test.drop('Id', axis=1)

trainy=df_train.loc[:,['SalePrice']]
df_train=df_train.drop('SalePrice', axis=1)



#check the missing term
df_train.isnull().sum()




sns.heatmap(df_train.corr(), vmax=.8, square=True);



corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(22, 8))
sns.heatmap(corrmat, vmax=.8, square=True,);



#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

##df_train[['LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt','BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF', 'GrLivArea', 'FullBath','GarageCars', 'GarageArea']]
##trainx=df_train.loc[:,[ 'OverallQual', 'OverallCond', 'YearBuilt', 'TotalBsmtSF', 'FullBath','GarageCars', ]]

##test=test.loc[:,[ 'OverallQual', 'OverallCond', 'YearBuilt', 'TotalBsmtSF', 'FullBath','GarageCars', ]]
test.isnull().sum()


df_train['GarageType'].fillna(df_train['GarageType'].mode()[0], inplace=True)



df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mode()[0], inplace=True)

df_train.isnull().sum()
df_train=pd.get_dummies(df_train)



df_train=df_train.drop('LotShape_IR1', axis=1)



test['GarageType'].fillna(test['GarageType'].mode()[0], inplace=True)
test['GarageArea'].fillna(test['GarageArea'].mode()[0], inplace=True)
test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mode()[0], inplace=True)
test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0], inplace=True)
test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0], inplace=True)
test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0], inplace=True)
test=pd.get_dummies(test)
        
        
        
        


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, trainy, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test = sc.fit_transform(test)



from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators = 100, random_state = 0)
rf.fit(X_train, y_train)
y_cv=rf.predict(X_test)
y_pred=rf.predict(test)



##PRINTING SCORE

print("RF SCORE" ,rf.score(X_test,y_test))




###### PERFORMANCE OPTIMIATION ##########
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator= rf, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()


sub['SalePrice']=y_pred
sub['Id']=test_original['Id']

pd.DataFrame(sub, columns=['SalePrice', 'Id']).to_csv('RF2.csv')



