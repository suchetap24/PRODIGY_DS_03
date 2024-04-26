from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
ds=pd.read_csv("C:\\Users\\SUCHETA P\\Downloads\\bank.csv")
#print(ds.head())
#print(ds.isnull().sum())
bins=[19,30,60,100]
ds['agegroup']=pd.cut(ds['age'],bins=bins,labels=['youngadult','middle-aged','oldaged'])

fig,axes=plt.subplots(2,2, figsize=(7,5))
sns.countplot(x='marital',hue='y',data=ds,ax=axes[0,0])
sns.countplot(x='education',hue='y',data=ds,palette='pink',ax=axes[0,1])
sns.countplot(x='job',hue='y',data=ds,ax=axes[1,0])
axes[1,0].tick_params(axis='x',rotation=90)
sns.countplot(x='agegroup',hue='y',data=ds,ax=axes[1,1])
plt.tight_layout()
plt.show()

label_encoder=LabelEncoder()
for col in ['job','marital','education','default','housing','loan','contact','month','poutcome','y']:
    ds[col]=label_encoder.fit_transform(ds[col])
print(ds.head()) 
ds=ds.drop(['agegroup'],axis=1)  
x=ds.drop(['y'],axis=1)
y=ds['y']
#print(x)
#print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print('x train shape:',x_train.shape)
print('x test shape:',x_test.shape)
print('y train shape:',y_train.shape)
print('y test shape:',y_test.shape)
dt=DecisionTreeClassifier(random_state=0)
dt.fit(x_train,y_train)
predict_y=dt.predict(x_test)
print('accuracy:',accuracy_score(y_test,predict_y))
print('precision score:',precision_score(y_test,predict_y,average='weighted'))
print('recall score:',recall_score(y_test,predict_y,average='weighted'))
print('f1 score:',f1_score(y_test,predict_y,average='weighted'))

prediction_df=pd.DataFrame({'Actual test':y_test,'Predicted':predict_y})
print(prediction_df.head(10))

subcribed=(predict_y==1).sum()
print('predicted to have subcribed:',subcribed)
subcribed_=(y_test==1).sum()
print('actually subcribed from dataset:',subcribed_)


