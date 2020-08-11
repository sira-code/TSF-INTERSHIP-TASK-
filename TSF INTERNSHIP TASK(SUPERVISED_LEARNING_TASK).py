#!/usr/bin/env python
# coding: utf-8

# # TASK 2:-

# # TO EXPLORE SUPERVISED MACHINE LEARNING

# In[2]:


#IN THIS SECTION, WE WILL SEE HOW THE PYTHON LIBRARIES FOR MACHINE LEARNING CAN BE USED TO IMPLWMENT 
#REGRESSION FUNCTIONS.
#WE WILL START WITH SIMPLE LINEAR REGRESSION INVOVING TWO VARIABLES.


# ## SIMPLE LINEAR REGRESSION 

# In[3]:


#SIMPLE LINEAR REGRESSION IS A METHOD TO HELP US IN UNDERSTAND THE RELATIONSHIP BETWEEN TWO VARIABLES .
#X:- Independent variable 
#Y:- Dependent variable 


# ### LINEAR FUNCTION

# In[4]:


# Y = a + bX
                # where, 
                         # a = intercept of the regression line 0 or the value of Y when X =0
                         # b = slope of the regression line or the velue with whivh Y changes when X increases
 # by 1 unit.


# ## TASK GIVEN:-
# #### IN THIS TASK(REGRESSION),we will predict the percentage of marks that a student is expected to score 
# #### based upon the no. of hours they studied.
# #### This is a simple linear regression task as it includes two variables.
# #### After training, predict the marks by a student who studies for 9.25 hours.

# In[5]:


print("START THE TASK ")


# ## IMPORTING THE NECESSARY LIBRARIES

# In[6]:


import numpy as np # useful for fundamental scientific computations.
import pandas as pd # for data analysis 
import matplotlib.pyplot as plt # library for data visualization
import seaborn as sns # it  is a Python data visualization library based on matplotlib
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
print("NECESSARY LIBRARIES ARE IMPORTED SUCCESSFULLY")


# ## LOADING THE DATASET 

# ### LOADING THE DATA IN CSV FORMAT  

# ### DATASET :-  
#      http://bit.ly/w-data

# In[7]:


student_data = pd.read_csv('student scores.csv')


# ## EXPLORING THE DATASET 

# In[8]:


print("CHEKING INFORMATION (DATA TYPES , MEMORY USAGE )")
student_data.info()


# In[9]:


print("SHAPE OF THE DATASET(NO.OF ROWS,NO.OF COLUMNS)")
student_data.shape


# In[10]:


print("PRINTING THE FIRST 5 ROWS")
print(student_data.head(5))
print("-----------------------------------")
print("PRINTING THE LAST 5 ROWS")
print(student_data.tail(5))


# In[11]:


student_data.head(25)


# In[12]:


print("CHECK THE NAN/NULL VALUES ")
student_data.isnull().sum()


# In[13]:


print("SUMMERY OF STATISTICS PERTAINING TO THE DATAFRAME COLUMNS")
student_data.describe()


# In[14]:


print("DATA TYPES FOR EACH COLUMN")
print(student_data.dtypes)


# In[15]:


print("PAIRWISE CORRELATION OF ALL COLUMNS IN THE DATAFRAME") 
student_data.corr()


# In[16]:


print("PLOT RECTANGULAR DATA AS A COLOR-ENCODED MATRIX")
corr = student_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True,cmap="Reds")
plt.title('CORRELATION HEATMAP',fontsize = 25)


# #### Now , we have seen the correlation of the data ,clearly there is a strong positive relationship between Scores and Hours

# ## PLOTTING THE DATA 

# ### VISUALIZING DISTRIBUTION OF VARIABLES THE GIVEN DATASET BY DIFFERENT PLOTS

# In[17]:


plt.figure(figsize=(10,10))
histogram = student_data[['Hours','Scores']]
histogram.hist()
plt.show()


# In[18]:


plt.figure(figsize=(10,10))
plt.scatter(student_data.Hours,student_data.Scores,color='green')
plt.title("HOURS VS PERCENTAGE")
plt.xlabel("HOURS")
plt.ylabel("PERCENTAGE SCORES")
plt.show()


# In[19]:


sns.lineplot(x="Hours",y="Scores",data=student_data)


# In[20]:


student_data.hist(figsize=(25,15),bins= 40)


# In[21]:


sns.barplot(x="Hours",y="Scores",data= student_data)


# In[22]:


sns.jointplot(x="Hours",y="Scores",data= student_data,kind="scatter")


# In[23]:


sns.pairplot(student_data)


# In[24]:


sns.boxplot(x="Hours",y="Scores",data= student_data,palette="rainbow")


# In[25]:


sns.violinplot(x="Hours",y="Scores",data= student_data,palette="rainbow")


# In[26]:


sns.stripplot(x="Hours",y="Scores",data= student_data)


# In[27]:


sns.swarmplot(x="Hours",y="Scores",data= student_data,palette="Set1",split=True)


# In[28]:


sns.catplot(x="Hours",y="Scores",data= student_data,kind="boxen")


# In[29]:


sns.lmplot(x="Hours",y="Scores",data= student_data,height=3,aspect=6)


# ## PRE-PROCESSING THE DATA 

# #### Data preprocessing in Machine Learningrefers to the technique of preparing (cleaning and organizing) the raw data to make it  suitable for a building and training Machine Learning models. 

# #### Fixing the target variable and segregating it from independent variables (X) in th Dataset

# In[30]:


#X IS THE FEATURE FOR LINEAR REGRESSION 
X = student_data.iloc[:,:-1].values
#y IS THE RESPONSE FOR LINEAR REGRESSION 
y = student_data.iloc[:,1].values


# In[31]:


print(X.shape)
print(y.shape)


# ## SPLITTING THE DATA

# ### Splitting the Dataset into Training set & Validation set

# In[32]:


from sklearn.model_selection import train_test_split


# ### Split the data around 30%-70% between testing and training stages

# In[33]:


X_train,X_valid,y_train,y_valid =train_test_split(X,y,train_size= 0.7,test_size=0.3,random_state=0)


# In[35]:


print(X_train.shape)
print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)
print(X_valid.shape)
print(y_valid.shape)


# In[36]:


plt.figure(figsize=(10,10))
plt.scatter(X_train,y_train,color='blue')
plt.title("HOURS VS PERCENTAGE")
plt.xlabel("HOURS")
plt.ylabel("PERCENTAGE SCORES")
plt.show()


# ## USING LINEAR REGRESSION MODEL( TRAINING THE LINEAR MODEL )

# ### Now, our data is split into train and test set ,let's import the linear regression from sklearn.linear_model

# In[37]:


from sklearn.linear_model import LinearRegression


# In[38]:


lr = LinearRegression() # create linearregression object 


# In[39]:


lr.fit(X_train,y_train)


# In[40]:


print("TRAINING IS OVER NOW")


# In[41]:


pred1=lr.predict(X_valid)
pred1


# In[42]:


print("THE COEFFICIENT IS:-",lr.coef_)
print("THE INTERCEPT IS:-",lr.intercept_)


# ## EVALUATING THE ACCURACY USING MEAN ABSOLUTE ERROR

# In[43]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[44]:


MAE1 = mean_absolute_error(pred1,y_valid)
RMSE1 = np.sqrt(mean_squared_error(y_valid,pred1))
print("MEAN ABSOLUTE ERROR:- ",MAE1.round(2))
print("ROOT MEAN SQUARED ERROR:- ",RMSE1.round(2))


# ## COMPARING THE ACTUAL VS PREDICTED VALUES 

# In[45]:


chk =pd.DataFrame({'ACTUAL' :y_valid})
chk.reset_index(drop=True ,inplace=True)
chk['PREDICTED']= pred1
chk['DEVIATION']=abs(chk['ACTUAL']-chk['PREDICTED'])
chk


# ## VISULAIZING THE DEVIATION IN ACTUAL VS PREDICTED VALUES 

# In[46]:


plt.figure(figsize=(8,8))
sns.regplot('PREDICTED','ACTUAL',data= chk,line_kws={'color':'blue'},scatter_kws={'color':'red'},marker='+')
plt.title('DEVIATION IN ACTUAL VS PREDICTED VALUES')


# ## USING LASSO REGULARIZATION MODEL

# ###   It is a regression analysis method that used to  performs both variable selection and regularization in order to enhance the prediction accuracy

# In[47]:


from sklearn.linear_model import Lasso


# In[48]:


ls = Lasso(alpha=1,random_state=0)


# In[49]:


ls.fit(X_train,y_train)


# ## MAKING PREDICTIONS

# In[50]:


pred2=ls.predict(X_valid)
pred2


# ## COMPARING ACTUAL VS PREDICTED VALUES 

# In[51]:


chk =pd.DataFrame({'ACTUAL' :y_valid})
chk.reset_index(drop=True ,inplace=True)
chk['PREDICTED']= pred2
chk['DEVIATION']=abs(chk['ACTUAL']-chk['PREDICTED'])
chk


# ## EVALUATING ACCURACY

# In[52]:


MAE2 = mean_absolute_error(pred2,y_valid)
RMSE2 = np.sqrt(mean_squared_error(y_valid,pred2))
print("MEAN ABSOLUTE ERROR:- ",MAE2.round(2))
print("ROOT MEAN SQUARED ERROR:- ",RMSE2.round(2))


# ## VISULAZING THE DEVIATION IN ACTUAL VS PREDICTED VALUES  

# In[53]:


plt.figure(figsize=(8,8))
sns.regplot('PREDICTED','ACTUAL',data= chk,line_kws={'color':'blue'},scatter_kws={'color':'red'},marker='+')
plt.title('DEVIATION IN ACTUAL VS PREDICTED VALUES(LASSO)')


# ## COMPARING BOTH MODELS (Actual vs predicted values) 

# In[54]:


final_data=pd.DataFrame()
errors=[MAE1,MAE2]
final_data['VALID']=y_valid
final_data['LINEAR REGRESSION']=pred1
final_data['LASSO']=pred2


# In[55]:


final_data.head()


# # PREDICTION
# ## ENTER THE NUMBER OF HOURS OF STUDY TO GET THE SCORE

# #### HOURS = 9.25 

# In[56]:


n=float(input())
result = lr.predict([[n]])
print("STUDYING FOR{} HOURS,THE EXPECTED SCORE WILL BE{}".format(n,result.round(2)))


# In[57]:


print("TASK COMPLETED SUCCESSFULLY")


# # STAY HOME AND KEEP LEARNING 
# ## : )
# 

# In[ ]:




