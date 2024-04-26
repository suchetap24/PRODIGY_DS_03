import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dataset=pd.read_csv("C:\\Users\\SUCHETA P\\Downloads\\twitter_validation.csv")
dataset=dataset.drop(['sentiment'],axis=1)
print(dataset.head(5))
#print(dataset['entity'].value_counts())
plt.figure(figsize=(6,4))
sns.histplot(x='entity',data=dataset)
plt.xticks(rotation=90,fontsize=5)
plt.show()
