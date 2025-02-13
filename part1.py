import numpy as np
from sklearn import datasets
import pandas as pd

data=datasets.load_iris()
names=data['target_names']

x=data['data']
y=data['target']

x=np.array(x,dtype=int)
data_=pd.DataFrame(x)
data_.columns=data.feature_names
data_['output']=y
#in order to check if any value is null in the data set
#print(data_.isna().sum())

#define entropy
def entropy(col):
    counts=np.unique(col,return_counts=True)
    N=float(col.shape[0])
    ent=0.0
    
    for ix in counts[1]:
        p=ix/N
        ent+=(-1.0*p*np.log2(p))

    return ent
def divide_data(x_data,key,val):
    x_left=pd.DataFrame([],columns=x_data.columns)
    x_right=pd.DataFrame([],columns=x_data.columns)

    for ix in range(x_data.shape[0]):
        val_=x_data[key].loc[ix]

        if(val_>val):
            x_right=x_right._append(x_data.loc[ix])
        else:
            x_left=x_left._append(x_data.loc[ix])

    return x_left,x_right


def information_gain(x_data,key,value):
    left,right=divide_data(x_data,key,value)

    if(left.shape[0]==0 or right.shape[0]==0):
        return -1000000
    
    l=float(left.shape[0])/x_data.shape[0]
    r=float(right.shape[0])/x_data.shape[0]
    i_gain=entropy(x_data.output)-(l*entropy(left.output)+r*entropy(right.output))

    return i_gain

def find_count(x_train):
    count=[]
    count.append(x_train[x_train['output']==0].shape[0])
    count.append(x_train[x_train['output']==1].shape[0])
    count.append(x_train[x_train['output']==2].shape[0])

    return count
class decisionTree:
    #constructor
    def __init__(self,depth=0,max_depth=5):
        self.left=None
        self.right=None
        self.key=None
        self.val=None
        self.count=None
        self.max_depth=max_depth
        self.depth=depth
        self.target=None
    
    def train(self,x_train,names):
        features=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
        info_gain=[]

        for ix in features:
            i_gain=information_gain(x_train,ix,x_train[ix].mean())
            info_gain.append(i_gain)

        self.key=features[np.argmax(info_gain)]
        self.val=x_train[self.key].mean()

        print("Level ",self.depth)
        self.count=find_count(x_train)

        cnt=0
        for i in range(len(self.count)):
            if(self.count[i]):
                print("Count of ",names[i],"=",self.count[i])
                cnt+=1

        print("Current Entropy = ",entropy(x_train['output']))
        if(cnt!=1):
            print("Splitting on Tree feature ",self.key, "with information gain ",max(info_gain))

        #split data
        data_left,data_right=divide_data(x_train,self.key,self.val)
        data_left=data_left.reset_index(drop=True)
        data_right=data_right.reset_index(drop=True)

        if cnt==1:
            if(x_train.output.mean()>=1.5):
                self.target=names[2]
            elif(x_train.output.mean()<=0.5):
                self.target=names[0]
            else:
                self.target=names[1]

            print("Reached leaf node")
            print()
            print()
            return
        
        if(self.depth>=self.max_depth):
            if(x_train['output'].mean()>=1.5):
                self.target=names[2]
            elif(x_train['output'].mean()<=0.5):
                self.target=names[0]
            else:
                self.target=names[1]

            print("max depth reach")
            print()
            print()
            return
        
        print()
        print()

        #recursion
        self.left=decisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.left.train(data_left,names)
        self.right=decisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.right.train(data_right,names)

        if(x_train.output.mean()>=1.5):
            self.target=names[2]
        elif(x_train.output.mean()<=0.5):
            self.target=names[0]
        else:
            self.target=names[1]

        return
    


dt=decisionTree()

dt.train(data_,names)

        
        