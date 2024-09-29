#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="https://drive.google.com/uc?export=view&id=1hDKusF04c0lNZA_qOShvWQrpImayVKz8"  width="1000" />
# </center>

# # Tugas Mandiri
# ---
# Tugas mandiri ini digunakan pada kegiatan Kursus Data Science yang merupakan pembekalan bagi mahasiswa Universitas Gunadarma untuk Skema Associate Data Scientist

# ### Pertemuan 4 - Semester 7

# 1. Buatlah model klasifikasi dengan machine learning dari dataset yang diberikan dengan ketentuan :
#     - Gunakan metode CRISP-DM secara terurut dan lengkap
#     - Gunakan algoritma linear regression, logistic regression, dan K-NN
# 
# 2. Dari ketiga algoritma yang anda pakai, algoritma yang manakah yang memiliki akurasi paling tinggi?

# In[1]:


## DATA UNDERSTANDING


# In[2]:


# Import Library
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Title
st.title("Data Science Modeling - Streamlit App")

# Load Data
st.header("Dataset")
df = pd.read_csv("Report.csv")  # Pastikan file ini tersedia di folder yang benar
st.write(df.head())

# Show data statistics
st.header("Dataset Statistics")
st.write(df.describe(include='all'))

# Outlier Detection
st.header("Outlier Detection")
q1 = df.select_dtypes(exclude='object').quantile(0.25)  # exclude should be a string, not a list
q3 = df.select_dtypes(exclude='object').quantile(0.75)
iqr = q3 - q1
batas_bawah = q1 - (1.5 * iqr)
batas_atas = q3 + (1.5 * iqr)

st.write("Lower Bound for Outliers:", batas_bawah)
st.write("Upper Bound for Outliers:", batas_atas)


# Histogram - Distribution of Total Amount
st.header("Distribution of Total Amount")
plt.figure(figsize=(10, 6))
sns.histplot(df['Total Amount'], bins=30, kde=True)
plt.title('Distribusi Total Amount')
plt.xlabel('Total Amount (IDR)')
plt.ylabel('Frequency')
st.pyplot(plt)

# Scatterplot - Total Amount vs Discounts
st.header("Scatter Plot - Total Amount vs Discounts")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Total Amount', y='Discounts', data=df)
plt.title('Korelasi Total Amount dan Discounts')
plt.xlabel('Total Amount (IDR)')
plt.ylabel('Discounts (IDR)')
st.pyplot(plt)

# Replace 'Datetime' with the correct column name, e.g., 'Date'
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert to datetime, handle errors

# Check for any invalid datetime conversions
if df['Date'].isnull().any():
    st.warning("Some Date values could not be converted and have been set to NaT. Please check the dataset.")

# Aggregate the transaction count per day
daily_df = df.groupby(df['Date'].dt.date).size().reset_index(name='Transaction Count')
daily_df.columns = ['Date', 'Transaction Count']

# Time Series plot - Transaction Count per Day
st.header("Time Series - Transaction Count per Day")
plt.figure(figsize=(12, 6))
plt.plot(daily_df['Date'], daily_df['Transaction Count'])
plt.title('Jumlah Transaksi Harian')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Transaksi')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(plt)


# Convert the correct date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Adjust to the actual column name

# Check if the 'Date' column was correctly converted
if df['Date'].isnull().any():
    st.warning("Some Date values could not be converted and have been set to NaT. Please check the dataset.")

# Extract the month from the Date column
df['Month'] = df['Date'].dt.month  # Update to the correct column name here

# Boxplot - Total Amount per Month
st.header("Boxplot - Total Amount per Month")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Month', y='Total Amount', data=df)
plt.title('Boxplot Total Amount per Bulan')
plt.xlabel('Bulan')
plt.ylabel('Total Amount (IDR)')
st.pyplot(plt)

# Pairplot
st.header("Pairplot of Total Amount and Discounts")
sns.pairplot(df[['Total Amount', 'Discounts']])
st.pyplot(plt)
