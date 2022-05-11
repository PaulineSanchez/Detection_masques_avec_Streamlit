# Détection de masques avec Streamlit

## Documents relatifs à ce projet
Pour ce projet, nous avons ré-utilisé le modèle qui se trouve dans *Detection_masques*.   
Dans ce repository, vous trouverez comme documents : 
- l'application Streamlit en python `app.py`
- le modèle sauvegardé : `modelmask`, avec comme sous-fichiers:
  - `saved_model.pb`
  - `keras_metadata.pb`
  - `variables.index`
  - `variables.data-00000-of-00001`
- le dossier `cascade` avec comme sous-fichier, le fichier XML:
  - `haarcascade_frontalface_default.xml`

  
## L'application Streamlit
L'application Streamlit est accessible en ligne [ici](https://share.streamlit.io/paulinesanchez/detection_masques_avec_streamlit/main/app.py).  
Cette application Streamlit demande d'abord de charger une image.  
Ensuite, une fois l'image chargée et affichée, il y a deux choix. Soit détecter les visages sur une image soit détecter et compter les visages mais aussi détecter si le visage en question porte ou non un masque.  
Si on choisit de détecter les visages, il y aura des carrés bleu qui entoureront tous les visages présents sur l'image.  
Si on choisit de compter les visages et de dire si ils portent ou non un masque, les visages portant un masque seront entourés de vert et les visages sans masque seront entourés en rouge. Il y aura également un compte des visages en dessous précisant combien portent un masque et combien ne portent pas un masque. En dessous, il y a un tableau qui répertorie tous les comptes et un bouton qui permet de télécharger ce tableau en Excel. Si on charge une autre image et qu'on choisit l'option de comptes & masques, les informations seront ajoutées au tableau.
