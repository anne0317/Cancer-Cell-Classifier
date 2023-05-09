import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import sklearn
from sklearn.svm import SVC
import itertools

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://c4.wallpaperflare.com/wallpaper/349/70/418/abstract-dark-black-background-digital-art-artwork-wallpaper-preview.jpg");
             background-attachment: fixed;
             opacity:0.0px;
             background-size: cover;
             color:white;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

st.title("CANCER CELL CLASSIFIER")

#st.title("Case study on dataset")

#Image
#st.image("confusion_matrix.png",width=500)


data=pd.read_csv("./cancer_cell_dataset.csv")

#ds=sns.load_dataset("diamonds")
#st.write("Shape of the dataset",data.shape)
#st.write(df)

menu=st.sidebar.radio("Menu",["Home","Prediction"])

model = pkl.load(open('rbfweights.pkl', 'rb'))

inputlist = []


if menu=="Home":
    st.title("Case study on dataset")
    data = data[pd.to_numeric(data['BareNuc'], errors="coerce").notnull()]

#After getting only numeric and non-null values, we can convert the type to int
    data["BareNuc"] = data["BareNuc"].astype('int64')


    X = np.asanyarray(data[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
    y = np.asanyarray(data['Class'].astype('int'))


    training_X, test_X, training_y, test_y = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    model_rbf = sklearn.svm.SVC(kernel='rbf')

    model_rbf.fit(training_X, training_y)
    y_hat = model_rbf.predict(test_X)


    st.header("F1-score of our model: {}".format(sklearn.metrics.f1_score(test_y, y_hat, average="weighted")))
    st.header("Jaccard index: {}".format(sklearn.metrics.jaccard_score(test_y, y_hat, pos_label=2)))
    st.write("Shape of the dataset",data.shape)

    st.header("Tabular Data of the dataset")
    if st.checkbox("Tabular Data"):
        st.table(data.head(699))
    st.header("Statistical Summary of the Dataset")
    if st.checkbox("Statistics"):
        st.table(data.describe())
    #if st.header("Correlation graph"):
    #    fig,ax=plt.subplots(1,1,figsize=(10,10))
    #    sns.heatmap(data.corr(),annot=True,cmap="coolwarm")
    #    st.pyplot(fig) 

        #1, 1, figsize=(10, 10)

    

    st.header("Correlational Matrix:")
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    sns.heatmap(data.corr(),annot=True,cmap="coolwarm")
    st.pyplot(fig)

    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Reds):
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    cf_matrix = sklearn.metrics.confusion_matrix(test_y, y_hat, labels=[2,4])
    np.set_printoptions(precision=2)

    html1 = """
    <div style="text-align:justify;font-family: font-family: 'EB Garamond', serif;">
      <br><p>The confusion matrix shows that the model correctly identified 91 out of 91 benign cases, resulting in a True Negative rate of 100%. However, the model incorrectly classified 16 out of 47 malignant cases as benign, resulting in a False Negative rate of 34%. This is a cause for concern, as it indicates that the model is not as effective in identifying malignant cases accurately. </p><br>
    </div>
      """
    st.markdown(html1, unsafe_allow_html=True)
    plt.figure()
    df=plot_confusion_matrix(cf_matrix, classes=['Benign(2)', 'Malignant(4)'], title='Confusion Matrix')
    st.pyplot(df) 

    st.header("CONCLUSION:")
    html3 = """
    <div style="text-align:justify;font-family: 'Ysabeau', sans-serif;">
      <br><h6>The cancer cell classifier that uses support vector machine (SVM) algorithm to predict if a cell is malignant or benign. The dataset used contains information on nine features, including cell size and shape, uniformity, and clump thickness, among others. The dataset was first cleaned by removing non-numeric values and then split into training and testing data. The SVM model with a radial basis function kernel was used to classify the cells. The model was then evaluated using the F1-score and Jaccard index, which both measure the accuracy of the model's predictions. A confusion matrix was used to visualize the performance of the model. The confusion matrix showed that the model correctly identified all 91 benign cases, resulting in a True Negative rate of 100%. However, the model incorrectly classified 16 out of 47 malignant cases as benign, resulting in a False Negative rate of 34%. This indicates that the model is not as effective in identifying malignant cases accurately. Overall, the SVM model is useful in predicting the classification of cancer cells as benign or malignant, with room for improvement in accurately identifying malignant cases. The code provides a user interface for inputting cell measurements and obtaining a prediction of the cell type.</h6><br>
    </div>
      """
    st.markdown(html3, unsafe_allow_html=True)

    st.set_option('deprecation.showPyplotGlobalUse',False)
    
    #Note 95% accuracy is lower than the RBF function-model.

    

st.sidebar.title("Your Information")

Name = st.sidebar.text_input("Full Name")

Contact_Number = st.sidebar.text_input("Contact Number")

Email_address = st.sidebar.text_input("Email address")

if not Name and Email_address:
    st.sidebar.warning("Please fill out your name and EmailID")

if Name and Contact_Number and Email_address:
    st.sidebar.success("Thanks!")

if menu=="Prediction":

    html2 = """
    <div style="text-align:justify;font-family: font-family: 'EB Garamond', serif;">
      <h3>Find out if the Tumour is Benign or Malignant <h3>
    </div>
      """
    st.markdown(html2, unsafe_allow_html=True)

    st.write('Fill in your measurements here')

    Clump = st.slider(
        'Clump Size', 0, 10, 1)
    st.write(Clump)
    inputlist.append(Clump)


    UnifSize = st.slider(
        'UnifSize', 0, 10, 1)
    st.write(UnifSize)
    inputlist.append(UnifSize)

    UnifShape = st.slider(
        'UnifShape', 0, 10, 1)
    st.write(UnifShape)
    inputlist.append(UnifShape)

    MargAdh = st.slider(
        'MargAdh', 0, 10, 1)
    st.write(MargAdh)
    inputlist.append(MargAdh)

    SingEpiSize = st.slider(
        'SingEpiSize', 0, 10, 1)
    st.write(SingEpiSize)
    inputlist.append(SingEpiSize)

    BareNuc = st.slider(
        'BareNuc', 0, 10, 1)
    st.write(BareNuc)
    inputlist.append(BareNuc)

    BlandChrom = st.slider(
        'BlandChrom', 0, 10, 1)
    st.write(BlandChrom)
    inputlist.append(BlandChrom)

    NormNucl = st.slider(
        'NormNucl', 0, 10, 1)
    st.write(NormNucl)
    inputlist.append(NormNucl)

    Mit = st.slider(
        'Mit', 0, 10, 1)
    st.write(Mit)
    inputlist.append(Mit)

    if st.button("Predict"):
        result = model.predict([inputlist])

        if result == 2:
            st.write('The tumour is Benign - it is not cancerous!')
        if result == 4:
            st.write('The tumour is Malignant - Consult a doctor now')

        st.write('For more info click here👇')

        st.write("[Benign vs Malignant Tumours](https://www.cancercenter.com/community/blog/2023/01/whats-the-difference-benign-vs-malignant-tumors)")

        f, axes = plt.subplots(1, 1, figsize=(10, 10))
        sns.heatmap(data.corr(), cmap='coolwarm', cbar=True)
        st.subheader("Exploratory Data Analysis on the Dataset: ")
        st.text("Correlation Between Numerical Features")
        st.pyplot(f)