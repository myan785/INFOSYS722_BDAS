
# coding: utf-8

# In[1]:


import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Iteration_4_BDAS').getOrCreate()


# In[2]:


import functools
# explicit function
def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)

df = spark.read.csv('Donation Retention Dataset.csv', inferSchema= True, header=True)

df1 = spark.read.csv('Donation Retention Dataset - 2.csv', inferSchema= True, header=True)
df = unionAll([df,df1])

df.printSchema()


# In[3]:


print((df.count(),len(df.columns)))


# In[4]:


df.describe().show()


# In[5]:


df.columns


# In[6]:


df.show()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt


# In[8]:


isCancelArr = np.array(df.select('is_cancel').collect())
plt.hist(isCancelArr)
plt.show()


# In[9]:


amoountArr = np.array(df.select('amount').collect())
amount_bins = [0,10,20,30,40,50,60,70,80,90,100,120]
amount_hist = plt.hist(amoountArr, bins=amount_bins)
amount_hist = plt.ylabel("count")
amount_hist = plt.xlabel('amount')
plt.show()


# In[10]:


ageArr = np.array(df.select('amount').collect())
age_bins = [15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
age_hist = plt.hist(ageArr, bins = age_bins)
age_hist = plt.ylabel("count")
age_hist = plt.xlabel("age")
plt.show()


# In[11]:


#Street
channel_1 = df.filter(df.sign_up_channel == "Street").count()
#D2D
channel_2 = df.filter(df.sign_up_channel == "Door to Door").count()
#Mall
channel_3 = df.filter(df.sign_up_channel == "Mall").count()
#Telephone
channel_4 = df.filter(df.sign_up_channel == "Telephone").count()

labels = 'Street', 'Door to Door', 'Mall', 'Telephone'
sizes = [channel_1, channel_2, channel_3, channel_4]
explode = (0.1,0,0,0)
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

plt.pie(sizes, explode = explode,labels = labels, colors = colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()


# In[12]:


#Male
gender_male = df.filter(df.gender == "Male").count()
#Female
gender_female = df.filter(df.gender == "Female").count()
#Unspecified
gender_unspecified =  df.filter(df.gender == "Unspecified").count()

labels = 'Male','Female','Unspecified'
sizes = [gender_male, gender_female, gender_unspecified]
explode = (0.1,0,0)
colors = ['gold','lightskyblue','yellowgreen']

plt.pie(sizes,explode = explode, labels = labels, colors = colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()


# In[13]:


#Weekly
freq_weekly = df.filter(df.donation_frequency == "Weekly").count()
#Montly
freq_monthly = df.filter(df.donation_frequency == "Monthly").count()
#Annually
freq_annually = df.filter(df.donation_frequency == "Annually").count()
#Other
freq_other = df.filter(df.donation_frequency == "Other").count()

labels = 'Mothly', 'Weekly,Annually & Other'
sizes = [freq_monthly,freq_annually + freq_weekly + freq_other]
explode = (0.1,0)
colors = ['gold', 'yellowgreen']

plt.pie(sizes,explode = explode, labels = labels, colors = colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()


# In[14]:


#Generate the histogram of is_credit_card
caradArr = np.array(df.select('is_credit_card').collect())
is_credit_hist = plt.hist(caradArr)
is_credit_hist = plt.xlabel("is_credit_card")
plt.show()

#Generate the histogram of is_instant_pay
instantArr = np.array(df.select('is_instant_pay').collect())
is_instant_pay_hist = plt.hist(instantArr)
is_instant_pay_hist = plt.xlabel("is_instant_pay")
plt.show()

#Generate the histogram of is_email_verified
emailVerifiedArr = np.array(df.select('is_email_verified').collect())
is_email_verified_hist = plt.hist(emailVerifiedArr)
is_email_verified_hist = plt.xlabel("is_email_verified")
plt.show()

#Generate the histogram of is_address_verified
addressVerifiedArr = np.array(df.select('is_address_verified').collect())
is_address_verified = plt.hist(addressVerifiedArr)
is_address_verified = plt.xlabel("is_address_verified")
plt.show()


# In[15]:


from pyspark.sql.functions import col,isnan, when, count
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
   ).show()


# In[16]:


print((df.count(),len(df.columns)))
df.columns


# In[17]:


df.select("occupation").distinct().show()
df = df.drop("occupation")
print((df.count(),len(df.columns)))


# In[18]:


df.select("donation_start_date").distinct().show()
df = df.drop("donation_start_date")
print((df.count(),len(df.columns)))


# In[19]:


df.select("sign_up_date").distinct().show()
df = df.drop("sign_up_date")
print((df.count(),len(df.columns)))
df.columns


# In[20]:


#Mean of the age
df = df.fillna({ 'age':37} )


# In[21]:


df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
   ).show()


# In[22]:


age_amount_cols = df.select("age","amount","is_cancel")

colors1='gold'
colors2='lightskyblue'

age_is_cancel = age_amount_cols.filter(age_amount_cols.is_cancel == 1).select("age","is_cancel")
age_not_cancel = age_amount_cols.filter(age_amount_cols.is_cancel == 0).select("age","is_cancel")

ageCancelArr0 = np.array(age_is_cancel.select('is_cancel').collect())
ageCancelArr1 = np.array(age_is_cancel.select('age').collect())

ageNotCancelArr0 = np.array(age_not_cancel.select('is_cancel').collect())
ageNotCancelArr1 = np.array(age_not_cancel.select('age').collect())

plt.xlabel('is_cancel')
plt.scatter(ageCancelArr0,ageCancelArr1,c=colors1, alpha=0.5, label='is_cancelled')
plt.scatter(ageNotCancelArr0,ageNotCancelArr1,c=colors2, alpha=0.5, label='not_cancelled')
plt.plot(color='#000000')
plt.legend()
plt.show()


# In[23]:


amount_is_cancel = age_amount_cols.filter(age_amount_cols.is_cancel == 1).select("amount","is_cancel")
amount_not_cancel = age_amount_cols.filter(age_amount_cols.is_cancel == 0).select("amount","is_cancel")

amountCancelArr0 = np.array(amount_is_cancel.select('is_cancel').collect())
amountCancelArr1 = np.array(amount_is_cancel.select('amount').collect())

amountNotCancelArr0 = np.array(amount_not_cancel.select('is_cancel').collect())
amountNotCancelArr1 = np.array(amount_not_cancel.select('amount').collect())

plt.xlabel('is_cancel')
plt.scatter(amountCancelArr0,amountCancelArr1,c=colors1, alpha=0.5, label='is_cancelled')
plt.scatter(amountNotCancelArr0,amountNotCancelArr1,c=colors2, alpha=0.5, label='not_cancelled')
plt.plot(color='#000000')
plt.legend()
plt.show()


# In[24]:


#Boxplot for age (Before removed)
ageArr = np.array(df.select('age').collect()) 
plt.boxplot(ageArr)
plt.xlabel('age')
plt.show()

#Boxplot for amount (Before removed)
amountArr = np.array(df.select('amount').collect())
plt.boxplot(amountArr)
plt.xlabel('amount')
plt.show()


# In[25]:


#Remove Outliers and Extremes
print("Before removed:",(df.count(),len(df.columns)))

#Remove age outliers and extremes
df = df.filter(df.age > 1)
df = df.filter(df.age <= 72)

#Remove amount outliers and extremes
df = df.filter(df.amount > 15)
df = df.filter(df.amount <= 45)

print("After removed:",(df.count(),len(df.columns)))


# In[26]:


#Boxplot for age (After removed)
ageArr = np.array(df.select('age').collect()) 
plt.boxplot(ageArr)
plt.xlabel('age')
plt.show()

#Boxplot for amount (After removed)
amountArr = np.array(df.select('amount').collect())
plt.boxplot(amountArr)
plt.xlabel('amount')
plt.show()


# In[27]:


df = df.withColumn("is_info_verified",col("is_email_verified")* col("is_address_verified"))
print("Dateset shape:",(df.count(),len(df.columns)))
df.printSchema()
df.select('is_info_verified').show()


# In[28]:


df.printSchema()


# In[29]:


from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import IntegerType

#Format Donation Frequency
df = df.withColumn('donation_frequency', regexp_replace('donation_frequency', 'Weekly', '1'))
df = df.withColumn('donation_frequency', regexp_replace('donation_frequency', 'Monthly', '2'))
df = df.withColumn('donation_frequency', regexp_replace('donation_frequency', 'Annually', '3'))
df = df.withColumn('donation_frequency', regexp_replace('donation_frequency', 'Other', '4'))

df = df.withColumn("donation_frequency", df["donation_frequency"].cast(IntegerType()))

#Format Signup Channel
df = df.withColumn('sign_up_channel', regexp_replace('sign_up_channel', 'Street', '1'))
df = df.withColumn('sign_up_channel', regexp_replace('sign_up_channel', 'Door to Door', '2'))
df = df.withColumn('sign_up_channel', regexp_replace('sign_up_channel', 'Mall', '3'))
df = df.withColumn('sign_up_channel', regexp_replace('sign_up_channel', 'Telephone', '4'))

df = df.withColumn("sign_up_channel", df["sign_up_channel"].cast(IntegerType()))

df.printSchema()


# In[30]:


df.show()


# In[31]:


#Merge Female & Unspecified to be Not Male
#Female count
print("Female count:",df.filter(df.gender == "Female").count())
#Unspecified count
print("Unspecified count:",df.filter(df.gender == "Unspecified").count())

df = df.withColumn('gender', regexp_replace('gender', 'Female', 'Not Male'))
df = df.withColumn('gender', regexp_replace('gender', 'Unspecified', 'Not Male'))

print("Not male count:",df.filter(df.gender == "Not Male").count())


# In[32]:


#Male
gender_male = df.filter(df.gender == "Male").count()
#Not Male
gender_not_male = df.filter(df.gender == "Not Male").count()

labels = 'Male','Not Male'
sizes = [gender_male, gender_not_male]
explode = (0.1,0)
colors = ['gold','lightskyblue']

plt.pie(sizes,explode = explode, labels = labels, colors = colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()


# In[33]:


#Change the datatype of gender be integer
#We will change it back to string later
df.select("gender").distinct().show()

df = df.withColumn('gender', regexp_replace('gender', 'Not Male', '0'))
df = df.withColumn('gender', regexp_replace('gender', 'Male', '1'))

df.select("gender").distinct().show()

df = df.withColumn("gender", df["gender"].cast(IntegerType()))

df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
   ).show()


# In[34]:


#Change to Pandas DataFrame
pandaDf = df.toPandas()

from imblearn.over_sampling import SMOTE
import pandas as pd

#over sample
without_sample_cols = pandaDf.drop(['is_credit_card', 'is_email_verified', 'is_instant_pay'], axis=1)

sample_cols = pandaDf['is_credit_card'] * 100 + pandaDf['is_email_verified'] * 10 + pandaDf['is_instant_pay']

sm = SMOTE(random_state=42)
without_sample_cols_res, sample_cols_res = sm.fit_resample(without_sample_cols, sample_cols)

#is_credit_card over sample col
sample_col_is_card = sample_cols_res.map(lambda x: int(x/100))
#is_email_verified over sample col
sample_col_is_email = sample_cols_res.map(lambda x: int(x%100/10))
#is_instant_pay over sample col
sample_col_is_instant = sample_cols_res.map(lambda x: (x%100%10))

pandaDf = pd.concat([pd.DataFrame(without_sample_cols_res), pd.DataFrame(
    sample_col_is_card
    , columns=['is_credit_card'])], axis=1)

pandaDf = pd.concat([pd.DataFrame(pandaDf), pd.DataFrame(
    sample_col_is_email
    , columns=['is_email_verified'])], axis=1)

pandaDf = pd.concat([pd.DataFrame(pandaDf), pd.DataFrame(
    sample_col_is_instant
    , columns=['is_instant_pay'])], axis=1)

without_sample_cols = pandaDf.drop('is_cancel', axis = 1)

sample_cols = pandaDf.is_cancel


without_sample_cols_res, sample_cols_res = sm.fit_resample(without_sample_cols, sample_cols)

pandaDf = pd.concat([pd.DataFrame(without_sample_cols_res), pd.DataFrame(
    sample_cols_res
    , columns=['is_cancel'])], axis=1)


# In[35]:


#draw the new hist
is_card_hist_new = plt.hist(pandaDf["is_credit_card"])
is_card_hist_new = plt.ylabel("count")
is_card_hist_new = plt.xlabel('is_credit_card')
plt.show()
print("is_credit_card(true) count:",pandaDf.is_credit_card.loc[pandaDf.is_credit_card == 1].count())
print("is_credit_card(false) count:",pandaDf.is_credit_card.loc[pandaDf.is_credit_card == 0].count())
print("is_credit_card(true) per%:",round(pandaDf.is_credit_card.loc[pandaDf.is_credit_card == 1].count() / pandaDf["is_credit_card"].count() * 100, 2) ,'%')
print("is_credit_card(false) per%:",round(pandaDf.is_credit_card.loc[pandaDf.is_credit_card == 0].count() / pandaDf["is_credit_card"].count() * 100, 2) ,'%')

#draw the new hist
is_email_hist_new = plt.hist(pandaDf["is_email_verified"])
is_email_hist_new = plt.ylabel("count")
is_email_hist_new = plt.xlabel('is_email_verified')
plt.show()
print("is_email_verified(true) count:",pandaDf.is_email_verified.loc[pandaDf.is_email_verified == 1].count())
print("is_email_verified(false) count:",pandaDf.is_email_verified.loc[pandaDf.is_email_verified == 0].count())
print("is_email_verified(true) per%:",round(pandaDf.is_email_verified.loc[pandaDf.is_email_verified == 1].count() / pandaDf["is_email_verified"].count() * 100, 2) ,'%')
print("is_email_verified(false) per%:",round(pandaDf.is_email_verified.loc[pandaDf.is_email_verified == 0].count() / pandaDf["is_email_verified"].count() * 100,2 ) ,'%')


#draw the new hist
is_instant_hist_new = plt.hist(pandaDf["is_instant_pay"])
is_instant_hist_new = plt.ylabel("count")
is_instant_hist_new = plt.xlabel('is_instant_pay')
plt.show()
print("is_instant_pay(true) count:",pandaDf.is_email_verified.loc[pandaDf.is_instant_pay == 1].count())
print("is_instant_pay(false) count:",pandaDf.is_email_verified.loc[pandaDf.is_instant_pay == 0].count())
print("is_instant_pay(true) per%:",round(pandaDf.is_email_verified.loc[pandaDf.is_instant_pay == 1].count() / pandaDf["is_instant_pay"].count() * 100, 2) ,'%')
print("is_instant_pay(false) per%:",round(pandaDf.is_email_verified.loc[pandaDf.is_instant_pay == 0].count() / pandaDf["is_instant_pay"].count() * 100, 2) ,'%')

is_cancel_hist_new = plt.hist(pandaDf["is_cancel"])
is_cancel_hist_new = plt.ylabel("count")
is_cancel_hist_new = plt.xlabel('is_cancel')
plt.show()


# In[36]:


#Credit card and instant pay
print("is_card & is_instant per%:",round(pandaDf[(pandaDf["is_credit_card"] == 1) 
& (pandaDf["is_instant_pay"] == 1)]["is_credit_card"].count() / pandaDf["is_credit_card"].count() * 100 ,2), "%")

#Creidt card and not instant pay
print("is_card & is_not_instant per%",round(pandaDf[(pandaDf["is_credit_card"] == 1)
& (pandaDf["is_instant_pay"] == 0)]["is_credit_card"].count() / pandaDf["is_credit_card"].count() * 100, 2), "%")
#Bank Account
print("is_bank & is_not_instant per%",round(pandaDf[(pandaDf["is_credit_card"] == 0) 
& (pandaDf["is_instant_pay"] == 0)]["is_credit_card"].count() / pandaDf["is_credit_card"].count() * 100, 2), "%")


# In[37]:


#Feature Importance
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

topFeatures = SelectKBest(score_func=f_classif, k=10)
X = pandaDf.drop('is_cancel', axis = 1)
Y = pandaDf.is_cancel
topFeatures.fit(X,Y)
scores = pd.DataFrame(topFeatures.scores_)
columns = pd.DataFrame(X.columns)
feature_scores = pd.concat([columns, scores], axis = 1)
print(feature_scores)


# In[38]:


#Change back to Spark Dataframe
df = spark.createDataFrame(pandaDf)
df = df.withColumn("age", df["age"].cast(IntegerType()))
df = df.withColumn("gender", df["gender"].cast(IntegerType()))
df = df.withColumn("donation_frequency", df["donation_frequency"].cast(IntegerType()))
df = df.withColumn("amount", df["amount"].cast(IntegerType()))
df = df.withColumn("is_address_verified", df["is_address_verified"].cast(IntegerType()))
df = df.withColumn("sign_up_channel", df["sign_up_channel"].cast(IntegerType()))
df = df.withColumn("is_info_verified", df["is_info_verified"].cast(IntegerType()))
df = df.withColumn("is_credit_card", df["is_credit_card"].cast(IntegerType()))
df = df.withColumn("is_email_verified", df["is_email_verified"].cast(IntegerType()))
df = df.withColumn("is_instant_pay", df["is_instant_pay"].cast(IntegerType()))
df = df.withColumn("is_cancel", df["is_cancel"].cast(IntegerType()))

df.printSchema()
df.describe().show()


# In[39]:


#Import Vectors and VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[40]:


#Visualise the columns to help with assembly
df.columns


# In[41]:


# Combine all features into one vector named features.
assembler = VectorAssembler(
  inputCols=['age',
 'gender',
 'donation_frequency',
 'amount',
 'is_address_verified',
 'sign_up_channel',
 'is_info_verified',
 'is_credit_card',
 'is_email_verified',
 'is_instant_pay'],
 outputCol="features")


# In[42]:


# Transform the data. 
output = assembler.transform(df)


# In[43]:


# Import the string indexer
from pyspark.ml.feature import StringIndexer

#Create Index
indexer = StringIndexer(inputCol="is_cancel", outputCol="IsCancelIndex")
output_fixed = indexer.fit(output).transform(output)
final_data = output_fixed.select("features",'IsCancelIndex')


# In[44]:


# Split the training and testing set.
train_data,test_data = final_data.randomSplit([0.8,0.2])


# In[54]:


# Import the DecisionTree 
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline

dt_gini = DecisionTreeClassifier(labelCol='IsCancelIndex',featuresCol='features', impurity='gini')
dt_entropy = DecisionTreeClassifier(labelCol='IsCancelIndex',featuresCol='features', impurity='entropy')


# In[72]:


#Import the LogsticRegression
from pyspark.ml.classification import LogisticRegression

lr_l1_iter1 = LogisticRegression(featuresCol='features',labelCol='IsCancelIndex'
                                 , family='binomial', elasticNetParam = 1, maxIter = 100)
lr_l2_iter1 = LogisticRegression(featuresCol='features',labelCol='IsCancelIndex'
                                 , family='binomial', elasticNetParam = 0, maxIter = 100)
lr_l2_iter2 = LogisticRegression(featuresCol='features',labelCol='IsCancelIndex'
                                 , family='binomial', elasticNetParam = 0, maxIter = 150)


# In[47]:


#Import the RandomForest
from pyspark.ml.classification import RandomForestClassifier

rf_gini = RandomForestClassifier(labelCol='IsCancelIndex',featuresCol='features', impurity='gini')
rf_entropy = RandomForestClassifier(labelCol='IsCancelIndex',featuresCol='features', impurity='entropy')


# In[95]:


#Import MultilayerPerception
from pyspark.ml.classification import MultilayerPerceptronClassifier

layers = [10, 4, 3, 2]

mlp_iter1 = MultilayerPerceptronClassifier(labelCol='IsCancelIndex',featuresCol='features'
                                           ,solver='l-bfgs',maxIter=100, layers=layers)
mlp_iter2 = MultilayerPerceptronClassifier(labelCol='IsCancelIndex',featuresCol='features'
                                           ,solver='l-bfgs',maxIter=200, layers=layers)


# In[65]:


#Train the DecisioTree Model

dt_gini_model = dt_gini.fit(train_data)
dt_entropy_model = dt_entropy.fit(train_data)

#Compare models

dt_gini_predictions = dt_gini_model.transform(test_data)
dt_entropy_predictions = dt_entropy_model.transform(test_data)

# Let's start off with multi classification.
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
     labelCol="IsCancelIndex", predictionCol="prediction", metricName="accuracy")

dt_gini_acc = evaluator.evaluate(dt_gini_predictions)
dt_entropy_acc = evaluator.evaluate(dt_entropy_predictions)

print("Decision Tree (gini) has an accuracy of {0:2.2f}%".format(dt_gini_acc*100))
print("Decision Tree (entropy) has an accuracy of {0:2.2f}%".format(dt_entropy_acc*100))


# In[74]:


#Train the LogisticRegression Model

lr_l1_iter1_model = lr_l1_iter1.fit(train_data)
lr_l2_iter1_model = lr_l2_iter1.fit(train_data)
lr_l2_iter2_model = lr_l2_iter2.fit(train_data)

lr_l1_iter1_predictions = lr_l1_iter1_model.transform(test_data)
lr_l2_iter1_predictions = lr_l2_iter1_model.transform(test_data)
lr_l2_iter2_predictions = lr_l2_iter2_model.transform(test_data)

lr_l1_iter1_acc = evaluator.evaluate(lr_l1_iter1_predictions)
lr_l2_iter1_acc = evaluator.evaluate(lr_l2_iter1_predictions)
lr_l2_iter2_acc = evaluator.evaluate(lr_l2_iter2_predictions)

print("Logistic Regression (l1 & iter=100) has an accuracy of {0:2.2f}%".format(lr_l1_iter1_acc*100))
print("Logistic Regression (l2 & iter=100) has an accuracy of {0:2.2f}%".format(lr_l1_iter1_acc*100))
print("Logistic Regression (l2 & iter=150) has an accuracy of {0:2.2f}%".format(lr_l1_iter1_acc*100))


# In[75]:


#Train the RandomForestModel

rf_gini_model = rf_gini.fit(train_data)
rf_entropy_model = rf_entropy.fit(train_data)

#Compare models

rf_gini_predictions = rf_gini_model.transform(test_data)
rf_entropy_predictions = rf_entropy_model.transform(test_data)

rf_gini_acc = evaluator.evaluate(rf_gini_predictions)
rf_entropy_acc = evaluator.evaluate(rf_entropy_predictions)

print("Random Forest Tree (gini) has an accuracy of {0:2.2f}%".format(rf_gini_acc*100))
print("Random Forest Tree (entropy) has an accuracy of {0:2.2f}%".format(rf_entropy_acc*100))


# In[96]:


#Train the MultilayerPerceptron

mlp_iter1_model = mlp_iter1.fit(train_data)
mlp_iter2_model = mlp_iter2.fit(train_data)

mlp_iter1_predictions = mlp_iter1_model.transform(test_data)
mlp_iter2_predictions = mlp_iter2_model.transform(test_data)

mlp_iter1_acc = evaluator.evaluate(mlp_iter1_predictions)
mlp_iter2_acc = evaluator.evaluate(mlp_iter2_predictions)

print("MLP (iter=100) has an accuracy of {0:2.2f}%".format(mlp_iter1_acc*100))
print("MLP (iter=200) has an accuracy of {0:2.2f}%".format(mlp_iter2_acc*100))


# In[100]:


rf_gini_model.featureImportances


# In[105]:


def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

ExtractFeatureImp(lr_l1_iter1_model.coefficients, dt_gini_predictions, "features").head(10)


# In[106]:


ExtractFeatureImp(rf_gini_model.featureImportances, dt_gini_predictions, "features").head(10)

