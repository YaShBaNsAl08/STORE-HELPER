#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')
# Load the data
sh = pd.read_csv("Mall_Customers.csv")

# Create the GUI
root = tk.Tk()
root.title("Store Helper")
root.geometry("800x600")

# Tab control
tab_control = ttk.Notebook(root)

# Data Exploration tab
data_exploration_tab = ttk.Frame(tab_control)
tab_control.add(data_exploration_tab, text="Data Exploration")

label_data_exploration = ttk.Label(data_exploration_tab, text="Data Exploration", font=("Helvetica", 16))
label_data_exploration.pack(pady=10)

# Box plots to check for outliers
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
sns.boxplot(x='Age', data=sh)
plt.subplot(3, 1, 2)
sns.boxplot(x='Annual Income (k$)', data=sh)
plt.subplot(3, 1, 3)
sns.boxplot(x='Spending Score (1-100)', data=sh)
plt.tight_layout()
canvas_data_exploration = FigureCanvasTkAgg(plt.gcf(), master=data_exploration_tab)
canvas_data_exploration.draw()
canvas_data_exploration.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
plt.close()

# Data Analysis tab
data_analysis_tab = ttk.Frame(tab_control)
tab_control.add(data_analysis_tab, text="Data Analysis")

label_data_analysis = ttk.Label(data_analysis_tab, text="Data Analysis", font=("Helvetica", 16))
label_data_analysis.pack(pady=10)

# Perform data analysis
sh.drop(["CustomerID"], axis=1, inplace=True)
x1 = sh.loc[:, ["Age", "Spending Score (1-100)"]].values
kmeans_age_spending = KMeans(n_clusters=4)
kmeans_age_spending.fit(x1)
clusters_age_spending = kmeans_age_spending.labels_

# Scatter plot of clusters
plt.figure(figsize=(8, 6))
plt.scatter(x1[:, 0], x1[:, 1], c=clusters_age_spending, cmap="viridis")
plt.scatter(
    kmeans_age_spending.cluster_centers_[:, 0],
    kmeans_age_spending.cluster_centers_[:, 1],
    color='red',
    marker='X',
    s=100,
    label='Cluster Centers'
)
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation based on Age and Spending Score")
plt.legend()
plt.tight_layout()
canvas_data_analysis = FigureCanvasTkAgg(plt.gcf(), master=data_analysis_tab)
canvas_data_analysis.draw()
canvas_data_analysis.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
plt.close()

# Data Visualization tab
data_visualization_tab = ttk.Frame(tab_control)
tab_control.add(data_visualization_tab, text="Data Visualization")

label_data_visualization = ttk.Label(data_visualization_tab, text="Data Visualization", font=("Helvetica", 16))
label_data_visualization.pack(pady=10)

# Data visualization
age1 = sh.Age[(sh.Age >= 18) & (sh.Age <= 25)]
age2 = sh.Age[(sh.Age >= 26) & (sh.Age <= 35)]
age3 = sh.Age[(sh.Age >= 36) & (sh.Age <= 45)]
age4 = sh.Age[(sh.Age >= 46) & (sh.Age <= 55)]
age5 = sh.Age[(sh.Age >= 56)]
agex = ["18-25", "26-35", "36-45", "46-55", "55+"]
agey = [len(age1.values), len(age2.values), len(age3.values), len(age4.values), len(age5.values)]
plt.figure(figsize=(15, 6))
sns.barplot(x=agex, y=agey, palette="mako")
plt.title("Number of Customers and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
canvas_data_visualization = FigureCanvasTkAgg(plt.gcf(), master=data_visualization_tab)
canvas_data_visualization.draw()
canvas_data_visualization.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
plt.close()

# Cluster Analysis tab
cluster_analysis_tab = ttk.Frame(tab_control)
tab_control.add(cluster_analysis_tab, text="Cluster Analysis")

label_cluster_analysis = ttk.Label(cluster_analysis_tab, text="Cluster Analysis", font=("Helvetica", 16))
label_cluster_analysis.pack(pady=10)

# Perform cluster analysis
x2 = sh.loc[:, ["Annual Income (k$)", "Spending Score (1-100)"]].values
kmeans_income_spending = KMeans(n_clusters=5)
kmeans_income_spending.fit(x2)
clusters_income_spending = kmeans_income_spending.labels_

# Scatter plot of clusters
plt.figure(figsize=(8, 6))
plt.scatter(x2[:, 0], x2[:, 1], c=clusters_income_spending, cmap="viridis")
plt.scatter(
    kmeans_income_spending.cluster_centers_[:, 0],
    kmeans_income_spending.cluster_centers_[:, 1],
    color='red',
    marker='X',
    s=100,
    label='Cluster Centers'
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation based on Annual Income and Spending Score")
plt.legend()
plt.tight_layout()
canvas_cluster_analysis = FigureCanvasTkAgg(plt.gcf(), master=cluster_analysis_tab)
canvas_cluster_analysis.draw()
canvas_cluster_analysis.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
plt.close()

# Age Analysis tab
age_analysis_tab = ttk.Frame(tab_control)
tab_control.add(age_analysis_tab, text="Age Analysis")

label_age_analysis = ttk.Label(age_analysis_tab, text="Age Analysis", font=("Helvetica", 16))
label_age_analysis.pack(pady=10)

# Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(sh["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
canvas_age_analysis = FigureCanvasTkAgg(plt.gcf(), master=age_analysis_tab)
canvas_age_analysis.draw()
canvas_age_analysis.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
plt.close()

tab6 = ttk.Frame(tab_control)
tab_control.add(tab6, text="Gender Analysis")

label6 = ttk.Label(tab6, text="Gender Analysis", font=("Helvetica", 16))
label6.pack(pady=10)

# Perform gender analysis
gender_counts = sh["Gender"].value_counts()
labels = gender_counts.index
sizes = gender_counts.values
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Gender Distribution")
plt.tight_layout()
canvas = FigureCanvasTkAgg(plt.gcf(), master=tab6)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
plt.close()

#tab 7 visualise the different clusters in the data
# Overall Analysis tab
overall_analysis_tab = ttk.Frame(tab_control)
tab_control.add(overall_analysis_tab, text="Overall Analysis")

label_overall_analysis = ttk.Label(overall_analysis_tab, text="Overall Analysis", font=("Helvetica", 16))
label_overall_analysis.pack(pady=10)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(x1)

# Assign cluster labels to the dataframe
sh["label"] = labels

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Define colors for each cluster
colors = ["blue", "red", "green", "black", "purple"]

# Scatter plot for each cluster
for i in range(5):
    ax.scatter(sh.Age[sh.label == i], sh["Annual Income (k$)"][sh.label == i],
               sh["Spending Score (1-100)"][sh.label == i], c=colors[i], s=60)

# Set labels and title
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
ax.set_title("Customer Segmentation based on Age, Annual Income, and Spending Score")

# Show the plot
canvas_overall_analysis = FigureCanvasTkAgg(fig, master=overall_analysis_tab)
canvas_overall_analysis.draw()
canvas_overall_analysis.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
plt.close()



# Pack the tab control
tab_control.pack(expand=1, fill="both")

root.mainloop()



# In[ ]:





# In[ ]:




