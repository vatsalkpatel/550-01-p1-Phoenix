"""Project 550-01-Dtree
Team members
1) Sean Pereira - Sean.Pereira@student.csulb.edu
2) Sushmitha Pasala - Sushmitha.Pasala@student.csulb.edu
3) Vatsal Patel - Vatsal.Patel01@student.csulb.edu"""

import subprocess

command = "pip install numpy" #command to be executed 
res = subprocess.call(command, shell = True)

command = "pip install pandas" #command to be executed
res = subprocess.call(command, shell = True)
"""
THE CODE
Created a Class containing all the required datatypes, and functions for building Decision Tree.
DATA TYPES
	df - DataFrame to save csv file content and to perform easy panda functions
	left - left subtree/classifier
	right - right subtree/classifier
	fkey - feature name on which it decides classification
	fval - Feature value which is used to divide data set for classifing
	depth - gives current tree depth
	max_depth - gives max discoverable depth
	target - label to be classified for
FUNCTIONS
init - Initializes all values for the object
	Parameters -
	Depth - current depth of tree
	Max_depth - Maximum depth
	Returns - None
label_output - Alters label for convinency of code
	Parameters - None
	Returns - None
processing_data - Converts multiple values to binary values and generating n number of columns from 1 column
	Parameters -
	Data - the DataFrame
	Returns -
	5 Dfs - 5 different dataframes for each feature vector
entropy - calculates entropy of desired feature
	Parameters -
	Col - columns of the record of which entropy has to be calculated
	Returns - entropy as float
information_gain - Calculates information gain from entropy of possible nodes
	Parameters -
	x_data - Feature Data set
	fkey - Feature name
	fval - Feature Value
	Returns -
	info_gain - returns information gain
divide_data - Divides the data set for classifying using maximum all info gains
	Parameters -
	x_data - Feature Data set
	fkey - Feature name
	fval - Feature Value
	Returns -
	X_left - left split Dataset
	X_right - right split Dataset
frequency_of_Output - gives frequency of the labels present in x_train datset
	Parameters -
	X_train - training Data set
	Returns -
	Max(frequent label)
train - calls different functions and trains the decision tree using training dataset ans its labels
	Parameters -
	X_train - training Data set
	Returns - None
predict - label to be classified for
	Parameters -
	Test - Testing Data set
	Returns -
Predicted class
	dataframe - label to be classified for
	Parameters - None
	Returns -
	whole Dataframe"""
import pandas as pd
import numpy as np
import random


class DecisionTree:
    def __init__(self,depth=0,max_depth=5):
        #Read the data from csv file and name the columns
        
        c=['White King file (column)','White King rank (row)','White Rook file','White Rook rank','Black King file','Black King rank','Output']
        self.df=pd.read_csv('550-p1-cset-krk-1.csv',header=None)
        self.df=self.df.rename({0:'White King file (column)',1:'White King rank (row)',2:'White Rook file',3:'White Rook rank',4:'Black King file',5:'Black King rank',6:'Output'}, axis=1)
        df0,df1,df2,df3,df4,df5=self.processing_data(self.df)
        self.label_output()
        self.df=pd.concat([df0,df1,df2,df3,df4,df5,self.df['Output']],axis=1)
        self.left=None # left branch
        self.right=None # right branch
        self.info_gain = 0.0
        self.fkey=None 
        self.fval=None
        self.depth=depth
        self.max_depth=max_depth
        self.target=None
        self.parent=None
        self.d1={17:'draw',0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten',11:'eleven',12:'twelve',13:'thirteen',14:'fourteen',15:'fifteen',16:'sixteen'}
        
    def label_output(self):
        #Converts Labels to int numbers
        self.d={'draw':17,'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11
          ,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16}
        #iterating over each column in dataframe
        for column in self.df:
            if column=='Output':
                s1=self.df[column].values
                for j,i in enumerate(s1):
                    s1[j]=self.d[i]
                break
        self.df=self.df.assign(Output=s1,inplace='True')

        
        
    def processing_data(self,data):
        # Labeling each data to 0-1, converting categorical to numerical data
        
        columns_text_0=['White King file (a)','White King file (b)','White King file (c)','White King file (d)','White King rank (e)','White King file (f)','White King rank (g)','White King file (h)']
        columns_data_0=['White King rank (1)','White King rank (2)','White King rank (3)','White King rank (4)','White King rank (5)','White King rank (6)','White King rank (7)','White King rank (8)']
        columns_text_1=['White Rook file (a)','White Rook file (b)','White Rook file (c)','White Rook file (d)','White Rook file (e)','White Rook file (f)','White Rook file (g)','White Rook file (h)']
        columns_data_1=['White Rook rank (1)','White Rook rank (2)','White Rook rank (3)','White Rook rank (4)','White Rook rank (5)','White Rook rank (6)','White Rook rank (7)','White Rook rank (8)']
        columns_text_2=['Black King file (a)','Black King file (b)','Black King file (c)','Black King file (d)','Black King file (e)','Black King file (f)','Black King file (g)','Black King file (h)']
        columns_data_2=['Black King rank (1)','Black King rank (2)','Black King rank (3)','Black King rank (4)','Black King rank (5)','Black King rank (6)','Black King rank (7)','Black King rank (8)']
        index=0
#         grabing each column to change it with binary columns
        for i in ['White King file (column)','White King rank (row)','White Rook file','White Rook rank','Black King file','Black King rank']:
            alphabets=[]
            numericals=[]
            for columndata in data[i]:
                letter=[0]*8
                numbers=[0]*8
                if not isinstance(columndata, int):
                    letter[ord(columndata)-ord('a')]=1
                    alphabets.append(letter)
                else:
                    numbers[ord(str(columndata))-ord('0')-1]=1
                    numericals.append(numbers)
#             Updating dataframe for the data manupilation
            if index==0:
                df0=pd.DataFrame(data=alphabets, columns=columns_text_0)
            if index==1:
                df1=pd.DataFrame(data=numericals, columns=columns_data_0)
            if index==2:
                df2=pd.DataFrame(data=alphabets, columns=columns_text_1)
            if index==3:
                df3=pd.DataFrame(data=numericals, columns=columns_data_1)
            if index==4:
                df4=pd.DataFrame(data=alphabets, columns=columns_text_2)
            if index==5:
                df5=pd.DataFrame(data=numericals, columns=columns_data_2)
            index+=1
        return (df0,df1,df2,df3,df4,df5)
# returns each column's dataframes
    
    def entropy(self,col):
        #calculates Entropy using log formula
        counts=np.unique(col,return_counts=True)
        ent=0.0
        for i in counts[1]:
#             probability
            p=i/col.shape[0]
#             calculating entropy
            ent+=(-1.0*p*np.log2(p))
            #heres the formula
        return ent
    #returns entropy
    
    def information_gain(self,x_data,fkey,fval):# calculates information gain from entropy of possible nodes
        right,left=self.divide_data(x_data,fkey,fval)
        l=float(left.shape[0])/x_data.shape[0]
        r=float(right.shape[0])/x_data.shape[0]
        if left.shape[0]==0 or right.shape[0]==0:
            return float("-inf")
        print("Average entropy:",(l*self.entropy(left.Output)+r*self.entropy(right.Output)))
#         finding information gain 
        i_gain=self.entropy(x_data.Output)-(l*self.entropy(left.Output)+r*self.entropy(right.Output))
        return i_gain#returns information gain of that probable node
    
    def divide_data(self,x_data,fkey,fval):
        #generates two dataframe as per the feature and its dividing value
        #fkey: Feature names 
        #fval: Feature values
        
        x_right=pd.DataFrame([],columns=x_data.columns)
        x_left=pd.DataFrame([],columns=x_data.columns)
        for i in range(x_data.shape[0]):
            val = x_data[fkey].loc[i]
            if val >= fval:
                x_right = x_right.append(x_data.iloc[i])
            else:
                x_left = x_left.append(x_data.iloc[i])
        return x_right,x_left#returns left and right dataframes
    
    def frequency_of_Output(self, x_train):
        
        self.dict={}
        for i in x_train:
            if i not in self.dict:
                self.dict[i]=1
            else:
                self.dict[i]+=1
        return max(self.dict, key= lambda d: self.dict[d])
        
    def train(self,x_train,parent): #calls different functions and trains the decision tree using training dataset ans its labels
        features=self.df.columns[:-1]
        info_gains=[]
        for i in features:
            i_gain=self.information_gain(x_train,i,0.5)
            info_gains.append(i_gain)
        
        self.parent=parent
        self.fkey=features[np.argmax(info_gains)]
        self.fval=0.5
        self.info_gain = max(info_gains)
        print("Splitting Tree ",self.fkey," with info gain ",max(info_gains),"Parent Node:",self.parent)
        data_right,data_left=self.divide_data(x_train,self.fkey,self.fval)
        data_right=data_right.reset_index(drop=True)
        data_left=data_left.reset_index(drop=True)
        if data_left.shape[0]==0 or data_right.shape[0]==0:
            
            self.target=self.d1[self.frequency_of_Output(x_train.Output)]
            return 
        if self.depth>=self.max_depth:
            
            self.target=self.d1[self.frequency_of_Output(x_train.Output)]
            return 
        self.left=DecisionTree(self.depth+1,self.max_depth)
        self.left.train(data_left,self.fkey)
        self.right=DecisionTree(self.depth+1,self.max_depth)
        self.right.train(data_right,self.fkey)

        self.target=self.d1[self.frequency_of_Output(x_train.Output)]
        return 
    
    def predict(self,test): #predicts the possible classification
#         compares if value is higher or lower than classifier value
        if test[self.fkey] > self.fval:
#         Checks if tree has right node
            if self.right is None:
#         returns the classified class
                return self.target
            return self.right.predict(test)
        if test[self.fkey] <= self.fval:
#         Checks if tree has left node
            if self.left is None:
#         returns the classified class
                return self.target
            return self.left.predict(test)
    def dataframe(self):# returns whole dataframe
        return self.df

    

        
#Creating Object of first Decision Tree
d=DecisionTree()



# Splitting Data Into training, test and validate :60,20,20
train_data, validate_data, test_data = np.split(d.dataframe().sample(frac=1,random_state=42), [int(.6*len(d.dataframe())), int(.8*len(d.dataframe()))])

#Reset Index to 0
train_data=train_data.reset_index(drop=True)
test_data=test_data.reset_index(drop=True)

# Building tree
d.train(train_data,None)

"""Decision tree into Dictionary
	decision_tree_algorithm - traverse through whole tree and creates dictionary
		Parameters -
			d - Decision Tree Class
			counter - Counter for tree level
		Returns -
			tree - tree in form of Dictionary"""

def decision_tree_algorithm(d, counter=0):         
    
    if (d.left is None) and (d.right is None) :
        return d.target
    fval = random.random()
    counter += 1
    # instantiate sub-tree
    question = "{} <= {} with info_gain = {}".format(d.fkey, fval,d.info_gain) # edit
    sub_tree = {question: []}

    # find answers (recursion)
    yes_answer = decision_tree_algorithm(d.left, counter)
    no_answer = decision_tree_algorithm(d.right, counter)

    # If the answers are the same, then there is no point in asking the qestion.
    # This could happen when the data is classified even though it is not pure
    # yet (min_samples or max_depth base cases).
    if yes_answer == no_answer:
        sub_tree = yes_answer
    else:
        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)

    return sub_tree
'''Printing the Decison Tree with Parent Nodes, Q-Nodes and Class of Leaf Nodes for Tree #1'''
tree = decision_tree_algorithm(d)
print(tree)

'''Train DataSet
Here we show the sneak peak of the train dataset containing 49 columns and 132 rows including output label

That was randomly selected from the whole dataset and this consists of 60% of the whole dataset.

And this consist some of the left over predictions from the previous training with probability of 3 times more than previous.'''
print(train_data.head())

'''Validation DataSet
Here we show the sneak peak of the validation dataset containing 49 columns and 44 rows including output label

That was randomly selected from the whole dataset and this consists of 20% of the whole dataset.'''
print(validate_data.head())

'''Holdout DataSet
Here we show the sneak peak of the test/holdout dataset containing 49 columns 44 rows including output label

That was randomly selected from the whole dataset and this consists of 20% of the whole dataset.'''
print(test_data.head())

'''BAGGING Function
	FUNCTIONS
		bagging_substitution - Adds falsily classified entries of holdout to training with 3 time probability for better trial of training decision tree
		Parameters -
			t_set - previous training set
			holdout_set - current holdout set
			d - Tree Object
		Returns -
			final_t_set - updated training set
			final_holdout_set - same holdout set
		accuracy - Checks for inaccuracy
			Parameters -
				test_set - Test Data set
				d - Tree Object
			Returns -
				returns accuracy'''

def bagging_susbstition(train_set, holdout_set,d): 
    
    # adds falsily classified entries of holdout to training with 3 time probability
    #for better trial of training decision tree

    
    Training_indexes = list(train_set.index)
    Testing_indexes = list(holdout_set.index)
    combined_set=Training_indexes
    combined_set.sort()
    
    
    final_train_set = []
    final_holdout_set = []
    
    incorrect_array=accuracy(d,holdout_set)[1]


    for i in incorrect_array:
        combined_set.append(i)
        combined_set.append(i)

    for _ in range(len(train_set)):
        index = random.randint(0, len(train_set) - 1)
        final_train_set.append(combined_set[index])


    # makining sure duplicate indixes are not in final_train_set
    for index_value in combined_set:
        if index_value not in final_holdout_set:
            final_holdout_set.append(index_value)
    
    for index_value in final_train_set:
        if index_value in final_holdout_set:
            final_holdout_set.remove(index_value)
        
            
    if len(final_holdout_set)> len(Testing_indexes):
        final_holdout_set=final_holdout_set[:44]
        
        

    return final_train_set, final_holdout_set

def accuracy(d,test_data): #checks for inaccuracy

    count=0
    incorrect=[]
    correct=[]
    old_data=test_data.index

    test_data=test_data.reset_index(drop=True)
    y_pred=[]

    for i in range(test_data.shape[0]):
        y_pred.append(d.predict(test_data.loc[i]))


    for i in range(len(y_pred)):
        if y_pred[i]== d.d1[test_data['Output'][i]]:
            count+=1
            correct.append(i)
        else:
            incorrect.append(i)
    
    new_data=[]
    for i in incorrect:
        new_data.append(old_data[i])  
    return count/len(test_data),new_data

print("Accuracy of 1st DTree:",accuracy(d,test_data)[0]*100,"%")
Training_Set, Holdout_Set = bagging_susbstition(train_data, test_data,d)

'''Converting Indexes to Dataframe Function
	FUNCTION
		convert_indices_to_DataFrame - Converts DataFrame from Numpy Array
	Parameters -
		Training_set - previous training set
			d - Tree Object
		Returns -
			v1 - Array of indices'''
def convert_indices_to_DataFrame(Training_Set,d): #converts dataframe from numpy array
    index1=[]
    Training_Set  
    d1=[]
    #iterates through the all rows of dataframe with index
    for i, j in d.dataframe().iterrows():
        if i in Training_Set:
            c1=Training_Set.count(i)
            for _ in range(c1):
                d1.append(d.dataframe()[i:i+1].values)
    v1=[]
    for i in d1:
        b1=[]
        for t in i:
            for r in t:
                # appending the rows
                b1.append(r)
        v1.append(b1)
    return v1

#d1=DecisionTree()
#obtaining training set from bagging
d1=DecisionTree()
Training_Set_d2 = pd.DataFrame(data= convert_indices_to_DataFrame(Training_Set,d1),columns=d1.dataframe().columns)
# obtaining holdout set from bagging
HoldOut_Set_d2 = pd.DataFrame(data=  convert_indices_to_DataFrame(Holdout_Set,d1),columns=d1.dataframe().columns)
d1.train(Training_Set_d2,None)

'''Printing the Decison Tree with Parent Nodes, Q-Nodes and Class of Leaf Nodes for Tree #2'''
tree = decision_tree_algorithm(d1)
print(tree)

'''Training set for decision tree #2'''
print(Training_Set_d2.shape)

'''Validate set for decision tree #2'''
print(validate_data.shape)

'''HoldOut set for decision tree #2'''
print(HoldOut_Set_d2.shape)

'''Accuracy of Decision Tree #2'''
print("Accuracy of 2nd DTree:",accuracy(d1,HoldOut_Set_d2 )[0]*100,"%")

'''Ensemble voting of Dtree #1 and Dtree #2
	FUNCTION
		ensemble_tree - Returns Best Tree and accuracy of Validation Data
		Parameters -
			tree1 - object of Tree1
			tree2 - object of Tree2
			validation_set - input of validation_set
		Returns -
			winning_tree, validating_accuracy - Tree with most accuracy and accuracy of validation_data'''

def ensemble_tree(tree1,tree2,validation_set):
    
    
    accuracy_tree1 = accuracy(tree1,validation_set)[0]*100
    accuracy_tree2 = accuracy(tree2,validation_set)[0]*100
    
    if accuracy_tree1 > accuracy_tree2:
        return 'Tree1',accuracy_tree1
    else:
        return 'Tree2',accuracy_tree2
    
winning_tree,  validating_accuracy= ensemble_tree(d,d1,validate_data)
print('The winning tree is:',winning_tree,"with accuracy:",validating_accuracy,"%")