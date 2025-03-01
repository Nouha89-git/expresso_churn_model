import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle


# Fonction pour charger le modèle
@st.cache_data
def load_model():
    return joblib.load('expresso_churn_model.pkl')


# Fonction pour charger les colonnes
@st.cache_data
def load_columns():
    with open('model_columns.pkl', 'rb') as file:
        return pickle.load(file)


# Configuration de la page
st.set_page_config(page_title="Prédiction de Churn Expresso", page_icon="📱")

# Titre de l'application
st.title("Prédiction de Churn pour Expresso")
st.write("Cette application prédit si un client est susceptible de quitter le service (churn).")

try:
    # Chargement du modèle et des colonnes
    model = load_model()
    model_columns = load_columns()

    # Création d'un formulaire pour la saisie des données
    with st.form("prediction_form"):
        st.subheader("Entrez les informations du client")

        # Ici, vous devrez ajouter des champs pour chaque caractéristique de votre modèle
        # Par exemple (à adapter selon vos colonnes réelles) :
        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input("Durée d'abonnement (mois)", min_value=0)
            monthly_charges = st.number_input("Frais mensuels", min_value=0.0)

        with col2:
            total_charges = st.number_input("Frais totaux", min_value=0.0)
            contract_type = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])

        # Bouton de soumission
        submit_button = st.form_submit_button(label="Prédire")

    # Prédiction lors de la soumission
    if submit_button:
        # Création d'un DataFrame avec les données saisies
        input_data = pd.DataFrame(
            [[tenure, monthly_charges, total_charges, contract_type]],
            columns=["tenure", "MonthlyCharges", "TotalCharges", "Contract"]
        )

        # Vous devrez ajuster ceci pour qu'il corresponde exactement à vos colonnes de modèle

        # Prédiction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Affichage du résultat
        st.subheader("Résultat de la prédiction")
        if prediction[0] == 1:
            st.error(f"Ce client est susceptible de quitter le service (Probabilité: {prediction_proba[0][1]:.2%})")
        else:
            st.success(f"Ce client est susceptible de rester (Probabilité: {prediction_proba[0][0]:.2%})")

        # Visualisation des facteurs importants (à adapter selon votre modèle)
        st.subheader("Facteurs influençant la prédiction")
        st.write("Graphique des facteurs importants à venir dans une future version.")

except Exception as e:
    st.error(f"Une erreur s'est produite: {e}")
    st.info(
        "Assurez-vous que les fichiers 'expresso_churn_model.pkl' et 'model_columns.pkl' existent dans le répertoire de l'application.")