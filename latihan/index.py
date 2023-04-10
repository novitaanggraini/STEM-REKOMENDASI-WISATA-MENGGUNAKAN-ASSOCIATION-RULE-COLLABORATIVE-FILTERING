import streamlit as st
from math import sqrt
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy

#import dataset
place = pd.read_excel('latihan/tourism_with_id.xlsx')
places = pd.read_excel('latihan/place lat.xlsx')
ratingA = pd.read_excel('latihan/rating CF.xlsx')
user = pd.read_excel('latihan/user_rating.xlsx')
rating = pd.read_excel('latihan/tourism_rating.xlsx')
userZ = pd.read_excel('latihan/user_rating.xlsx')



#Navbar
nav = st.sidebar.selectbox("Menu",["Beranda","Association Rule","Collaborative Filtering"])
if nav == "Beranda":
    st.title("Sistem Rekomendasi")
    st.text("Berikut ini adalah data yang digunakan dalam sistem rekomendasi ini:")
    
    #data tempat
    st.subheader("Data Tempat")
    st.write("Pada data tempat terdapat data tempat wisata sebanyak 437. data tempat wisata tersebut berasal dari 5 kota besar yaitu : Surabaya,Bandung, Yogyakarta, Jakarta dan Semarang")
    st.map(places)
    st.dataframe(place)

    #data rating A
    st.subheader("Data rating Association rule")
    st.write("Pada data rating terdapat 10.000 rating tempat wisata dari user")
    st.dataframe(ratingA)

    #data rating CF
    st.subheader("Data rating Collaborative filtering")
    st.write("Pada data rating terdapat 10.000 rating tempat wisata dari user")
    st.dataframe(rating)

    #data User
    st.subheader("Data User")
    st.write("Pada data User terdapat 300 data user yang memberikan rating")
    st.dataframe(userZ)

if nav == "Association Rule":
    
    st.title("Association Rule")
    st.write("Metode ini untuk menentukan pola tempat yang dikunjungi")

    # Matrix User - item
    st.subheader("Matrix User-item")   
    main = ratingA.pivot_table(index='User_Id',columns='Place_Name',values='Rating')
    main.fillna(0, inplace=True)
    st.dataframe(main)

    # Encode Values
    st.subheader("Encode Values")
    st.write("Merubah semua nilai positif menjadi 1 dan selain itu menjadi 0")
    #converting all positive values to 1 and everything else to 0
    def my_encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    my_place_sets = main.applymap(my_encode_units)
    st.dataframe(my_place_sets)

    #Generating Frequent itemsets
    my_frequent_itemsets = apriori(my_place_sets, min_support=0.03, use_colnames=True)

    #generating rules
    my_rules = association_rules(my_frequent_itemsets, metric="lift", min_threshold=1)
    st.subheader("Rules")
    st.write("Pola Kunjungan Wisata yang didapatkan dari metode ini adalah sebagai berikut:")
    #st.dataframe(rules)
    st.table(my_rules.applymap(lambda x: tuple(x) if isinstance(x, frozenset) else x ))

if nav == "Collaborative Filtering":
    st.title("Collaborative Filtering")
    st.write("Metode ini memberikan rekomendasi berdasarkan preferensi anda")

     #read data 
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rating,reader=reader)

    #train data
    trainset= data.build_full_trainset()
    model = SVD()
    model.fit(trainset)

    #data semua tempat wisata
    all_place = rating.Place_Id.unique()

    #Tempat yang sudah di rating user
    user_id=st.number_input("Masukkan User ID",1,300)  

    if st.button("Submit"):
        st.success(f"Rekomendasi Tempat Wisata untuk Anda adalah")   
        visited = rating[rating.User_Id == user_id].Place_Id

        #tempat yng belum di rating/ di kunjungi
        not_visited = [Place_Id for Place_Id in all_place if Place_Id not in visited]
    
        #Prediksi Rating Tempat yang belum di kunjungi/di rating
        pred_rating = [model.predict(user_id,Place_Id).est for Place_Id in not_visited]

        #rekomendasi
        st.subheader("Rekomendasi Tempat Wisata")
        result = pd.DataFrame({"Place_Id": not_visited,"pred_rating":pred_rating})
        result = pd.merge(result,places,on='Place_Id')
        result.sort_values("pred_rating",ascending=False, inplace= True)
        st.map(result[:10])
        st.dataframe(result[:10])

        #menghitung MAE
        #full training set
        trainset1= data.build_full_trainset()
        testset1 = trainset1.build_anti_testset()
        predictions1 = model.test(testset1)
        st.write(f"MAE full training set = ",accuracy.mae(predictions1, verbose=True))

        #70/30 train/test split
        trainSet, testSet = train_test_split(data, test_size=.30, random_state=1)
        predictions = model.test(testSet)
        st.write(f"MAE 70/30 train/test = ",accuracy.mae(predictions, verbose=True)) 

 