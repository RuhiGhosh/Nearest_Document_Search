# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:59:21 2019

@author: nspac
"""
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import tkinter
from tkinter import *

from PIL import Image, ImageTk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.neighbors import KDTree
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import tkinter.scrolledtext as tkscrolled
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def getSentiment(story):
    df=pd.DataFrame(columns=['Topics','News'])
    df['Topics'] = [1]
    df['News'] = story
    df['News'] = df['News'].str.replace('[^\w\s]','')
    df['News'] = df['News'].str.replace('\d+','')
    df['News'] = df['News'].apply(stopwords)
    df['News'] = df['News'].apply(stem_sentences)
    #print(type(df['News']), df['News'])
    
    analyser = SentimentIntensityAnalyzer()
    sent=(analyser.polarity_scores(df['News'][0]))
    li=list(sent.keys())
    lv=list(sent.values())
    ly=[]
    for i in lv:
        print(i*100)
        ly.append(i*100)
    ly
    lif=[]
    lif=li[:3]
    #print(lif)
    liv=[]
    liv=ly[:3]
    #print(liv)
    #plt.bar(li,ly)
    #plt.show()
    liff=['Depressing','Neutral ','Pleasant']
    colors = ['red', 'gold', 'green']
    explode = (0, 0, 0)  # explode a slice if required
    
    plt.pie(liv, explode=explode, labels=liff, colors=colors,
            autopct='%1.1f%%', shadow=True)
    
    #draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
    fig = plt.gcf()
    fig.patch.set_facecolor([0.5,0.5,0.5])
    pic = fig.gca().add_artist(centre_circle)
    
    
     #Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.show() 
    
    path = (rpath + "\\tempUI\\Sentitemp.png")
    fig.savefig(path)
    return(path)

rpath=os.getcwd()
print(rpath)
if not os.path.exists(rpath+"\\tempUI"):
    os.makedirs(rpath+"\\tempUI")

#lpath="C:\\Users\\nspac\\Desktop\\Term 3\\Capstone Project 2\\Last and final\\NewData"
lpath="G:\\Analytics_Praxis\\Capstone\\NewData"
x={}
for i in os.listdir(lpath):
    new_path=lpath+'\\'+i
    for j in os.listdir(new_path):    
        f=open(new_path+'\\'+j,'r',encoding='ansi',errors='ignore')
        data=f.read()
        x[i+'_'+j.split('.')[0]]=" ".join(data.split('\n'))
        f.close()
data_frame=pd.DataFrame(columns=['Topics','News'])
data_frame['Topics']=x.keys()
data_frame['News']=x.values()

data_frame.reset_index(drop=True, inplace=True)
lemmatizer=WordNetLemmatizer()
import numpy as np
from nltk.corpus import stopwords
data_frame['News2'] = data_frame['News'].str.replace('[^\w\s]','')
data_frame['News2'] = data_frame['News2'].str.replace('\d+', '')


sw = set(stopwords.words('english'))


def stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)


data_frame["News3"] = data_frame["News2"].apply(stopwords)


lemmatizer=WordNetLemmatizer()


# doing the lemmatization
def stem_sentences(sentence):
    tokens = sentence.split()
    lemma_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemma_tokens)

data_frame['News4'] = data_frame['News3'].apply(stem_sentences)
data_frame.head()


# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data_frame["News4"])


# Tf-Idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
data_frame['tfidf']=list(X_train_tfidf.toarray())


Business_data_frame=data_frame[data_frame.Topics.str.contains("Business")]
Business_data_frame.reset_index(drop=True, inplace=True)

Entertainment_data_frame=data_frame[data_frame.Topics.str.contains("Entertainment")]
Entertainment_data_frame.reset_index(drop=True, inplace=True)

Politics_data_frame=data_frame[data_frame.Topics.str.contains("Politics")]
Politics_data_frame.reset_index(drop=True, inplace=True)

Science_data_frame=data_frame[data_frame.Topics.str.contains("Science")]
Science_data_frame.reset_index(drop=True, inplace=True)

Sports_data_frame=data_frame[data_frame.Topics.str.contains("Sports")]
Sports_data_frame.reset_index(drop=True, inplace=True)

Education_data_frame=data_frame[data_frame.Topics.str.contains("Education")]
Education_data_frame.reset_index(drop=True, inplace=True)


import time
from sklearn.neighbors import NearestNeighbors
start_time = time.time()
# Model for Business
model1=NearestNeighbors(metric='cosine', algorithm='brute')
model1.fit(Business_data_frame['tfidf'].tolist())
# Model for Entertainment
model2=NearestNeighbors(metric='cosine', algorithm='brute')
model2.fit(Entertainment_data_frame['tfidf'].tolist())
# Model for Politics
model3=NearestNeighbors(metric='cosine', algorithm='brute')
model3.fit(Politics_data_frame['tfidf'].tolist())
# Model for Science
model4=NearestNeighbors(metric='cosine', algorithm='brute')
model4.fit(Science_data_frame['tfidf'].tolist())
# Model for Sports
model5=NearestNeighbors(metric='cosine', algorithm='brute')
model5.fit(Sports_data_frame['tfidf'].tolist())
#print('It took %s seconds' %(time.time()-start_time)) 

model6=NearestNeighbors(metric='cosine', algorithm='brute')
model6.fit(Education_data_frame['tfidf'].tolist())


#print(model1.kneighbors(Business_data_frame["tfidf"][Business_data_frame['Topics']=='Business_'+'2'].tolist(),n_neighbors=3))
distances,indices1 = model1.kneighbors(Business_data_frame["tfidf"][Business_data_frame['Topics']=='Business_9'].tolist(),n_neighbors=3)
distances,indices2 = model2.kneighbors(Entertainment_data_frame["tfidf"][Entertainment_data_frame['Topics']=='Entertainment_3'].tolist(),n_neighbors=3)
distances,indices3 = model3.kneighbors(Politics_data_frame["tfidf"][Politics_data_frame['Topics']=='Politics_4'].tolist(),n_neighbors=3)
distances,indices4 = model4.kneighbors(Science_data_frame["tfidf"][Science_data_frame['Topics']=='Science_1'].tolist(),n_neighbors=3)
distances,indices5 = model5.kneighbors(Sports_data_frame["tfidf"][Sports_data_frame['Topics']=='Sports_8'].tolist(),n_neighbors=3)
distances,indices6 = model6.kneighbors(Education_data_frame["tfidf"][Education_data_frame['Topics']=='Education_3'].tolist(),n_neighbors=3)


def news_window():
    
    def readRecommended():
        cat = catagories[catValue.get()-1][0]
        #NewsNum = str(v.get())
        listSize = similar_list.size()
        if listSize <1: 
            NewsNum = str(v.get())
        else:
            NewsNum = similar_list.get(ACTIVE)[0]
        print("readListSize: ", listSize)
        print("read News",NewsNum)    
        f = open(rpath + "\\"+cat+"\\"+NewsNum+".txt",'r')
        #print(v.get())
        story = f.read()
        f.close()
        
        news_story.delete(1.0, END)
        news_story.insert(1.0, story)

        wordcloudN = WordCloud().generate(story)
        wcFileN = rpath + "\\tempUI\\tempr.png"
        wordcloudN.to_file(wcFileN)
        image2 = Image.open(wcFileN)
        Wcloud_photoN = ImageTk.PhotoImage(image2)
        wordCloud_label.config(image = Wcloud_photoN)
        wordCloud_label.image = Wcloud_photoN
        #wordCloud_label.grid(row = 0, column = 3)
        #newWindow.update_idletasks()
        
        print(story)
        
        path = getSentiment(story)
        image4 = Image.open(path)
        senti_photo = ImageTk.PhotoImage(image4)    
        sentiPlot_label = tkinter.Label(newWindow, height =300,width = 450, image = senti_photo, text = 'Sentiment Plot')
        sentiPlot_label.image = senti_photo
        sentiPlot_label.grid(row = 3, column = 1)
        sentiPlot_label.config(compound = 'top')
        
        
        return 1


    
    def showSummary():
        
        def closeNewsSummary():
            summaryWindow.destroy()
            return 1
        
        parser = PlaintextParser.from_string(news_story.get(1.0,END),Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, 2)
        
        summaryWindow = tkinter.Tk()
        summaryWindow.geometry("200x250")
        summaryWindow.title("News Summary")

        msgBox = tkinter.Message(summaryWindow, text = summary, width=200)
        msgBox.grid(row = 0, column = 0)
        
        closeButton = tkinter.Button(summaryWindow, text = "Close", height = 2, width = 7, command = closeNewsSummary)
        closeButton.grid(row = 1, column = 0)
        return 1
    
    
    def recommend():
        
        cat = catagories[catValue.get()-1][0]
        #NewsNum = str(v.get())
        listSizeR = similar_list.size()
        print("list Size R: ", listSizeR)
#        listContent = similar_list.get(ACTIVE)
#        print("list Content: ", listContent)
        if listSizeR == 0:
            NewsNum = str(v.get())
        else:
            NewsNum = similar_list.get(ACTIVE)[0]
        print("#News: ",NewsNum)
        
        if cat == "Business":
            distances,indices = model1.kneighbors(Business_data_frame["tfidf"][Business_data_frame['Topics']=='Business_'+NewsNum].tolist(),n_neighbors=4)
        if cat == "Entertainment":
            distances,indices = model2.kneighbors(Entertainment_data_frame["tfidf"][Entertainment_data_frame['Topics']=='Entertainment_'+NewsNum].tolist(),n_neighbors=4)
        if cat == "Politics":
            distances,indices = model3.kneighbors(Politics_data_frame["tfidf"][Politics_data_frame['Topics']=='Politics_'+NewsNum].tolist(),n_neighbors=4)
        if cat == "Science":
            distances,indices = model4.kneighbors(Science_data_frame["tfidf"][Science_data_frame['Topics']=='Science_'+NewsNum].tolist(),n_neighbors=4)
        if cat == "Sports":
            distances,indices = model5.kneighbors(Sports_data_frame["tfidf"][Sports_data_frame['Topics']=='Sports_'+NewsNum].tolist(),n_neighbors=4)
        if cat == "Education":
            distances,indices = model6.kneighbors(Education_data_frame["tfidf"][Education_data_frame['Topics']=='Education_'+NewsNum].tolist(),n_neighbors=4)
        
        print("3-NN",indices.tolist())
        print(type(indices))
        print(indices[0].tolist())
        
        similar_list.delete(0, END)
        
        newsHeader = ['']*4
        i = 0
        for index in indices[0]:
            if index == 0:
                index = '10'
            f = open(rpath+"\\"+folder+"\\"+str(index)+".txt",'r')
            #story = f.read() 
            #print(index)
            newsHeader[i] =str(index) + ': '+f.readline().strip()
            f.close()
            similar_list.insert(END, newsHeader[i])
            i = i + 1
#        for index in indices[0]:
#            similar_list.insert(END,index)
        
        return 1
    
    def closeWindow():
        newWindow.destroy()
        return 1
    
    newWindow = tkinter.Toplevel()
    newWindow.geometry("1120x700+160+25")
    newWindow.title(catagories[catValue.get()-1][0])
    #print(sub)
    folder = catagories[catValue.get()-1][0]
    f = open(rpath+"\\"+folder+"\\"+str(v.get())+".txt",'r')
    print(v.get())
    story = f.read()
    f.close()
    
    wordcloud = WordCloud().generate(story)
    wcFile = rpath + "\\tempUI\\temp.png"
    wordcloud.to_file(wcFile)
    #f = open("C:\\Users\\nspac\\Desktop\\Term 3\\Capstone Project 2\\Last and final\\NewData\\"+catagories[selRadio-1][0]+"\\"+str(i+1)+".txt",'r')
    #News[i] = (f.readline().strip(),i+1)
    #f.close()


    #scrollbar = Scrollbar(newWindow, orient=VERTICAL)
    #listbox = Listbox(frame, yscrollcommand=scrollbar.set)
    
    news_story = tkscrolled.ScrolledText(master = newWindow,height = 25,width = 70)#,yscrollcommand=scrollbar.set )
    #scrollbar.config(command=news_story.yview)
    news_story.grid(row = 3, column = 0)#, columnspan  = 8)
    news_story.insert(1.0,story)
    #news_story.pack(side=LEFT, fill=BOTH, expand=1)
    #Remove_text = story from news_story
    
    
    Summarisation_Button = tkinter.Button(newWindow, text = 'Summarisation',command = showSummary)
    Summarisation_Button.grid(row = 0, column = 0)

    Recommendation_Button = tkinter.Button(newWindow, text = 'Recommendation',command = recommend)
    Recommendation_Button.grid(row = 0, column = 1)    
    
    similar_list = tkinter.Listbox(newWindow, selectmode='BROWSE', height = 8,width =70)
    similar_list.grid(row = 1, column = 1)
    
    Read_button = tkinter.Button(newWindow,text = 'Read', command = readRecommended)
    Read_button.grid(row =  2, column = 1) 
    
    image1 = Image.open(wcFile)
    Wcloud_photo = ImageTk.PhotoImage(image1)
    #print(type(photo))
    wordCloud_label = tkinter.Label(newWindow, height = 210,width = 450, image = Wcloud_photo, text = 'Word Cloud')
    wordCloud_label.image = Wcloud_photo
    wordCloud_label.grid(row = 1, column = 0)
    wordCloud_label.config(compound = 'top')
    #wordCloud_label.create_image(photo)
    
    path = getSentiment(story)
    image4 = Image.open(path)
    senti_photo = ImageTk.PhotoImage(image4)    
    sentiPlot_label = tkinter.Label(newWindow, height =350,width = 450, image = senti_photo, text = 'Sentiment Plot')
    sentiPlot_label.image = senti_photo
    sentiPlot_label.grid(row = 3, column = 1)
    sentiPlot_label.config(compound = 'top')
    
    closeWindow_Button = tkinter.Button(newWindow, text = 'Close',command = closeWindow, height = 2, width = 12)
    closeWindow_Button.grid(row = 4, column = 1)
    return 1


def display_news():
    News=[(1,1)]*10
    selRadio = catValue.get()
    print(selRadio)
    for i in range(0,10):
            f = open(rpath+"\\"+catagories[selRadio-1][0]+"\\"+str(i+1)+".txt",'r')
            News[i] = (f.readline().strip(),i+1)
            f.close()
    print(News)
    for text, mode in News:
        b = Radiobutton(window, text=text,variable=v,indicatoron = 0, value=mode, command = news_window)
        b.grid(row = mode*2, column = 1 ,sticky = EW)
    return 1

window = tkinter.Tk()
window.geometry("800x270+300+150")
window.title("News")

v = IntVar()
catagories = [("Business",1),("Sports",2),("Education",3),("Entertainment",4),("Politics",5),("Science",6)]
catValue = IntVar()
for cat, val in catagories:
    cat_radioBtn = Radiobutton(window, text = cat, variable = catValue, value = val, indicatoron = 1,command = display_news,anchor = W)
    cat_radioBtn.grid(row = val*2, column = 0, sticky = EW)

mainloop()