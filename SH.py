import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

sh = pd.read_csv("Mall_Customers.csv")

# Printing first 5 records of the data frame
print(sh.head())

# Checking for the count of rows and columns
print(sh.shape)

#describe the bytes in the fixed size block
print(sh.dtypes)

# Looking at the datatypes on the columns
print(sh.info())

# Checking for null values
print(sh.isnull().sum())

# Plotting box plots to check for outliers
plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
sns.boxplot(x='Age',data=sh)
plt.subplot(3,1,2)
sns.boxplot(x='Annual Income (k$)',data=sh)
plt.subplot(3,1,3)
sns.boxplot(x='Spending Score (1-100)',data=sh)
#plt.show()

# Statistical information 
print(sh.describe())

sh.drop(["CustomerID"],axis = 1, inplace = True)
print(sh.drop)
#print(sh.head())

plt.figure(1, figsize = (15,6))
n = 0
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.histplot(sh[x] , bins = 20)
    plt.title('Histplot of {}'.format(x))
plt.show()

#representing number of male and female
plt.figure(figsize = (15,5))
#sns.countplot(y = 'Gender', data = sh)
#plt.show()
sns.set_theme(style="whitegrid")
ax1=sns.countplot(x='Gender',data=sh)
plt.show()

age1 = sh.Age[(sh.Age >=18) & (sh.Age <= 25)]
age2 = sh.Age[(sh.Age >=26) & (sh.Age <= 35)]
age3 = sh.Age[(sh.Age >=36) & (sh.Age <= 45)]
age4 = sh.Age[(sh.Age >=46) & (sh.Age <= 55)]
age5 = sh.Age[(sh.Age >=56)]
agex = ["18-25","26-35","36-45","46-55","55+"]
agey = [len(age1.values),len(age2.values),len(age3.values),len(age4.values),len(age5.values)]
plt.figure(figsize = (15,6))
sns.barplot(x = agex , y = agey , palette = "mako")
plt.title("Number of Customers and ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()

sns.relplot(x = "Annual Income (k$)" , y = "Spending Score (1-100)" , data = sh)

ss1 = sh["Spending Score (1-100)"][(sh["Spending Score (1-100)"] >= 1) & (sh["Spending Score (1-100)"] <= 20)]
ss2 = sh["Spending Score (1-100)"][(sh["Spending Score (1-100)"] >= 21) & (sh["Spending Score (1-100)"] <= 40)]
ss3 = sh["Spending Score (1-100)"][(sh["Spending Score (1-100)"] >= 41) & (sh["Spending Score (1-100)"] <= 60)]
ss4 = sh["Spending Score (1-100)"][(sh["Spending Score (1-100)"] >= 61) & (sh["Spending Score (1-100)"] <= 80)]
ss5 = sh["Spending Score (1-100)"][(sh["Spending Score (1-100)"] >= 81) & (sh["Spending Score (1-100)"] <= 100)]
ssx = ["1-20","21-40","41-60","61-80","81-100"]
ssy = [len(ss1.values),len(ss2.values),len(ss3.values),len(ss4.values),len(ss5.values)]
plt.figure(figsize = (15,6))
sns.barplot(x = ssx , y = ssy , palette = "rocket")
plt.title("Spending Score")
plt.xlabel("Score")
plt.ylabel("Number of Customer Having the Score")
plt.show()

ai1 = sh["Annual Income (k$)"][(sh["Annual Income (k$)"] >=0) & (sh["Annual Income (k$)"] <= 30)]
ai2 = sh["Annual Income (k$)"][(sh["Annual Income (k$)"] >=31) & (sh["Annual Income (k$)"] <= 60)]
ai3 = sh["Annual Income (k$)"][(sh["Annual Income (k$)"] >=61) & (sh["Annual Income (k$)"] <= 90)]
ai4 = sh["Annual Income (k$)"][(sh["Annual Income (k$)"] >=91) & (sh["Annual Income (k$)"] <= 120)]
ai5 = sh["Annual Income (k$)"][(sh["Annual Income (k$)"] >=121) & (sh["Annual Income (k$)"] <= 150)]
aix = ["$ 0 - 30,000","$ 30,001-60,000","$ 60,001-90,000","$ 90,001-1,20,000","$ 1,20,001-1,50,000"]
aiy = [len(ai1.values),len(ai2.values),len(ai3.values),len(ai4.values),len(ai5.values)]
plt.figure(figsize = (15,6))
sns.barplot(x = aix , y = aiy , palette = "Spectral")
plt.title("Annual Income")
plt.xlabel("Income")
plt.ylabel("Number of Customer")
plt.show()

x1 = sh.loc[:,["Age" , "Spending Score (1-100)"]].values
from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    kmeans = KMeans(n_clusters = k, init = "k-means++")
    kmeans.fit(x1)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth = 2, color = "red" , marker = "8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()
plt.figure(figsize=(15,6))
kmeans = KMeans(n_clusters = 4)
label = kmeans.fit_predict(x1)
print(label)

print(kmeans.cluster_centers_)

plt.scatter(x1[:,0], x1[:,1], c = kmeans.labels_, cmap = "rainbow")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color = "black")
plt.title("Clusters of Cuctomers")
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.show()

x2 = sh.loc[:,["Annual Income (k$)" , "Spending Score (1-100)"]].values
from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    kmeans = KMeans(n_clusters = k, init = "k-means++")
    kmeans.fit(x2)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth = 2, color = "red" , marker = "8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters =5)
label = kmeans.fit_predict(x2)

print(label)

print(kmeans.cluster_centers_)
plt.figure(figsize=(15,6))
plt.scatter(x2[:,0], x2[:,1], c = kmeans.labels_, cmap = "rainbow")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color = "black")
plt.title("Clusters of Cuctomers")
plt.xlabel("Annual Icome (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

x3 = sh.iloc[:,1:]
wcss=[]
for k in range(1,11):
    kmeans = KMeans(n_clusters = k, init = "k-means++")
    kmeans.fit(x3)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(15,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth = 2, color = "red" , marker = "8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters = 5)
label = kmeans.fit_predict(x3)
print(label)

print(kmeans.cluster_centers_)

clusters = kmeans.fit_predict(x3)
sh["label"] = clusters

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (40,20))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(sh.Age[sh.label == 0], sh["Annual Income (k$)"][sh.label == 0],sh["Spending Score (1-100)"][sh.label ==0], c ="blue", s = 60)
ax.scatter(sh.Age[sh.label == 1], sh["Annual Income (k$)"][sh.label == 1],sh["Spending Score (1-100)"][sh.label ==1], c ="red", s = 60)
ax.scatter(sh.Age[sh.label == 2], sh["Annual Income (k$)"][sh.label == 2],sh["Spending Score (1-100)"][sh.label ==2], c ="green", s = 60)
ax.scatter(sh.Age[sh.label == 3], sh["Annual Income (k$)"][sh.label == 3],sh["Spending Score (1-100)"][sh.label ==3], c ="black", s = 60)
ax.scatter(sh.Age[sh.label == 4], sh["Annual Income (k$)"][sh.label == 4],sh["Spending Score (1-100)"][sh.label ==4], c ="purple", s = 60)
ax.view_init(30,185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel("Spending score (1-100)")
plt.show()
