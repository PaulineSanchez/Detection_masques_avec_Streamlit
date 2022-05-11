import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
import io
import xlsxwriter 
import datetime
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from keras import datasets, layers, models


st.set_page_config(layout="centered")   
if "liste_date" not in st.session_state:
    st.session_state["liste_date"] = []
if "liste_heure" not in st.session_state:
    st.session_state["liste_heure"] = []
if "nombre_personnes" not in st.session_state:
    st.session_state["nombre_personnes"] = []
if "nombre_personnes_avec_masque" not in st.session_state:
    st.session_state["nombre_personnes_avec_masque"] = []
if "nombre_personnes_sans_masque" not in st.session_state:
    st.session_state["nombre_personnes_sans_masque"] = []



cascade_path = "./cascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)


def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    compteur = 0
    for (x, y, w, h) in faces:
        compteur = compteur + 1 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(img, (x, y - 20), (x + w, y), (0,0, 255), -1)
        cv2.putText(img, "Personne: " + str(compteur),(x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255) ) 
    return img, faces



def count_mask(our_image):
    id2label = {0: "Avec Masque", 1: "Sans Masque"}
    compteur_avec_masque = 0
    compteur_sans_masque = 0
    m_m = tf.keras.models.load_model("modelmask")
    new_img = np.array(our_image.convert('RGB'))
    src = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    rect = cascade.detectMultiScale(gray, 1.1, 4)
    if len(rect) > 0:
        for i,[x, y, w, h] in enumerate(rect):
            img_trimmed = src[y:y + h, x:x + w]
            img_trimmed = cv2.resize(img_trimmed,(224,224))     # resize image to match model's expected sizing
            img_trimmed = img_trimmed.reshape(1,224,224,3)
            img_trimmed = np.array(img_trimmed)
            prediction = int(m_m.predict(img_trimmed)[0][0])
            if prediction == 1:
                compteur_sans_masque = compteur_sans_masque + 1 
                cv2.rectangle(src, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.rectangle(src, (x, y - 20), (x + w, y), (255,0,0), -1)
                cv2.putText(src, f"Personne: {i+1} {id2label[prediction]}",(x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0) ) 
            if prediction == 0:
                compteur_avec_masque = compteur_avec_masque + 1
                cv2.rectangle(src, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(src, (x, y - 20), (x + w, y), (0, 255, 0), -1)
                cv2.putText(src, f"Personne: {i+1} {id2label[prediction]}",(x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0) )
        return src, rect, compteur_avec_masque, compteur_sans_masque
    else : 
        return src, rect, compteur_avec_masque, compteur_sans_masque
   

def main():
    st.title(" :mag: Image detection")
    st.subheader(" :flashlight: This app detects faces and counts people (and specifies if they have a mask or not :mask:)")
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if image_file is not None:
            our_image = Image.open(image_file)
            st.subheader("Your image")
            st.image(our_image)

            detect = st.checkbox("Detect faces")
            count = st.checkbox("Count people & tell if they have a mask")
            
            # Detect faces
            if detect:
                while True :
                    st.subheader("Detect faces :mag:")
                    result_img, result_faces = detect_faces(our_image)
                    if len(result_faces) == 0 : 
                        st.write("Aucune personne détectée")
                        break
                    if len(result_faces) is not None:  
                        st.image(result_img)
                        break
               
                
            # Count faces
            if count:
                while True :
                    st.subheader("Count people and tell if they have a mask :mask:") 
                    result_img, result_faces, compteur_avec_masque, compteur_sans_masque  = count_mask(our_image)
                    if len(result_faces) == 0 :
                        st.write("Impossible aucune personne détectée")
                        break
                        
                    if len(result_faces) is not None :
                        buffer = io.BytesIO()
                        e = datetime.datetime.now()
                        date = "%s/%s/%s" % (e.day, e.month, e.year)
                        heure = "%s:%s:%s" % (e.hour, e.minute, e.second)
                        
                        st.session_state["liste_date"].append(date)
                        st.session_state["liste_heure"].append(heure)
                        st.session_state["nombre_personnes"].append(len(result_faces))
                        st.session_state["nombre_personnes_avec_masque"].append(compteur_avec_masque)
                        st.session_state["nombre_personnes_sans_masque"].append(compteur_sans_masque)
                        df1 = pd.DataFrame({'Nombre de personne(s)': st.session_state["nombre_personnes"], 'Date:': st.session_state["liste_date"], 'Heure': st.session_state["liste_heure"], 'Personne(s) avec masque' : st.session_state["nombre_personnes_avec_masque"], 'Personne(s) sans masque' : st.session_state["nombre_personnes_sans_masque"]})
                        st.image(result_img)
                        if len(result_faces) < 2 : 
                            if compteur_avec_masque > 0 :
                                st.success("{} personne a été trouvée.".format(len(result_faces)) + "Il y a {} personne avec un masque".format(compteur_avec_masque))
                            if compteur_sans_masque > 0 :
                                st.success("{} personne a été trouvée.".format(len(result_faces)) + "Il y a {} personne sans masque".format(compteur_sans_masque))
                        else :
                            if compteur_avec_masque > 1 and compteur_sans_masque > 1 :
                                st.success("{} personnes ont été trouvées.".format(len(result_faces)) + "Il y a {} personnes avec un masque".format(compteur_avec_masque) + " et il y a {} personnes sans masque".format(compteur_sans_masque))
                            if compteur_avec_masque > 1 and compteur_sans_masque < 2 :
                                st.success("{} personnes ont été trouvées.".format(len(result_faces)) + "Il y a {} personnes avec un masque".format(compteur_avec_masque) + " et il y a {} personne sans masque".format(compteur_sans_masque))
                            if compteur_avec_masque < 2 and compteur_sans_masque < 1 :
                                st.success("{} personnes ont été trouvées.".format(len(result_faces)) + "Il y a {} personne avec un masque".format(compteur_avec_masque) + " et il y a {} personnes sans masque".format(compteur_sans_masque))
                            if compteur_avec_masque < 2 and compteur_sans_masque < 2 :
                                st.success("{} personnes ont été trouvées.".format(len(result_faces)) + "Il y a {} personne avec un masque".format(compteur_avec_masque) + " et il y a {} personne sans masque".format(compteur_sans_masque))
                        

                        # Create a Pandas Excel writer using XlsxWriter as the engine.
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            # Write each dataframe to a different worksheet.
                            df1.to_excel(writer, sheet_name='Sheet1')

                            # Close the Pandas Excel writer and output the Excel file to the buffer
                            writer.save()
                            st.dataframe(df1)
                            st.download_button(
                            label="Download Excel worksheets",
                            data=buffer,
                            file_name="detection.xlsx",
                            mime="application/vnd.ms-excel")
                            
                            st.write("Si vous souhaitez relancer la page et reset les données dans le tableau pressez : CTRL + SHIFT + R")
                        break
                             

 
if __name__ == '__main__':
    main()


