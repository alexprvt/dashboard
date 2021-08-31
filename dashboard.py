"""
@author: Alexandre
"""

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

from graphs import default_gauge
from graphs import plotly_waterfall
from graphs import plot_categ_bars
from graphs import plot_bars
from graphs import plot_distribution
from graphs import plot_repartition
from functions import predict_modified
from functions import transform_shap
  
URL = 'http://localhost:8501'
path = ''
# path = 'C:\\Users\\Alexandre\\Desktop\\Dashboard\\'
#  streamlit run C:/Users/Alexandre/Desktop/Dashboard_Test/dashboard_test.py



def main():
    
    #--------------------------------------------------------------------------
    ### Fonctions d'obtention d'informations des prêts dans la base de données
    #--------------------------------------------------------------------------
    
    @st.cache
    def get_ids():
        #chargement de la table
        score_table = pd.read_csv(path + 'score_table.csv')
        ids = score_table.SK_ID_CURR.values[:1000]
        return ids
    
    
    @st.cache
    def get_score(sk_id):
        #chargement de la table
        score_table = pd.read_csv(path + 'score_table.csv')
        score = score_table.set_index('SK_ID_CURR').loc[sk_id].SCORE
        return score
    
    
    @st.cache
    def get_target(sk_id):
        #chargement de la table
        score_table = pd.read_csv(path + 'score_table.csv')
        target = score_table.set_index('SK_ID_CURR').loc[sk_id].TARGET
        if target == 0:
            decision = 'Crédit accordé'
        else:
            decision = 'Crédit refusé' 
        return decision, target
    
    @st.cache
    def get_shap_values(sk_id):
        #chargement de la table des shap_values
        shap_data = pd.read_csv(path+'shap_data_1000.csv')
        shap_series = shap_data.set_index('Feature')[str(sk_id)]
        return shap_series
    
    @st.cache
    def load_data():
        #chargement des données brutes
        data = pd.read_csv(path +'dashboard_data.csv')
        data = data.set_index('SK_ID_CURR')
        return data
    
    @st.cache
    def load_data_full():
        #chargement des données (avec label encoding)
        data = pd.read_csv(path +'data_full.csv')
        data = data.set_index('SK_ID_CURR')
        return data
    
    @st.cache
    def get_value(sk_id, var):
        #chargement des données
        data = load_data()
        value = data.loc[sk_id][var]
        return value
    
    @st.cache
    def get_value_full(sk_id, var):
        #chargement des données
        data = load_data_full()
        value = data.loc[sk_id][var]
        return value
    #--------------------------------------------------------------------------
    ### Configuration de la page streamlit
    #--------------------------------------------------------------------------
    
    # Chargement du logo à placer dans l'onglet de la page
    logo_AP = Image.open(path+'logo_AP.PNG')       
    
    st.set_page_config(
         # permet à l'app web de prendre tout l'écran
        layout="wide", 
        # titre de la page                             
        page_title="Dashoard: Prêt à Dépenser",  
        # configuration du logo de la page
        page_icon=logo_AP,
        # impose l'ouverture de labarre d'options au lancement de l'app                       
        initial_sidebar_state="expanded",            
        )
    
    # Ajout du titre du Dashboard intéractif
    st.title("__*Dashboard intéractif: \
             compréhension de la capacité de remboursement d'un crédit*__")
     # Ajout du  lien vers le repository Github du projet
    st.markdown('Alexandre PRÉVOT - \
              *[Lien vers le dépôt Github](https://github.com/alexprvt)*')
    
    #Ajout d'une barre d'option latérale
    st.sidebar.title("Options d'affichage")
    
    
    #--------------------------------------------------------------------------
    ### Sélection de l'ID du prêt parmi les identifiants de la base de données
    #--------------------------------------------------------------------------
    
    IDS = get_ids()
    select_id = st.sidebar.selectbox('Identifiant du crédit: ', IDS)
    
    
    #--------------------------------------------------------------------------
    ### Sélection du type d'affichage
    #--------------------------------------------------------------------------
    
    menu=['Capacité de remboursement',
          'Informations relatives au client',
          'Modification des paramètres']
    
    radio = st.sidebar.radio("Renseignements", menu)
    
    
    #--------------------------------------------------------------------------
    ### Renseignements sur la capacité de remboursement du prêt
    #--------------------------------------------------------------------------
    
    if radio == 'Capacité de remboursement':
        st.header("__Prédiction de capacité à rembourser le prêt__")
        
        
        #Calcul de la probabilité de difficulté de paiement
        default_proba = int(100*(1-get_score(select_id)))
        fig = default_gauge(default_proba)
        
        #Séparation de l'espace en deux colonnes
        left_col, right_col = st.beta_columns(2)
        
        #Jauge de probabilité de difficulté de paiement
        left_col.plotly_chart(fig, use_container_width=True)
        
        seuil=31
        
        if default_proba <= seuil:
            decision = "Crédit accordé"
            emoji = "white_check_mark"
            
        else:
            decision = "Crédit refusé"
            emoji = "no_entry_sign"
          
        right_col.header(f"***Décision:*** {decision} :{emoji}:")
        
        right_col.write("_Lorsque la probabilité qu'il y ait un défaut de paiement \
                 dépasse 31%, l'algorithme prédit que le prêt ne doit pas être\
                 accordé. Ce seuil a été déterminé afin de maximiser le profit\
                 moyen par demandeur de crédit. Après application de ce \
                 seuil, 65% des prêts testés ont été correctement classifiés._")
        
        #Permet le choix du seuil de probabilité de défaut admissible
        seuil = right_col.slider(
                "Seuil d'acceptabilité",
                min_value=1,
                max_value=99,
                value=31,
                step=1,
                help="Probabilité maximale du risque de défaut de paiement \
                acceptable pour accorder le prêt"
                )
        
        expander = right_col.beta_expander("En savoir plus sur la méthode de prédiction")
        expander.write("La prédiction de probabilité de défaut de paiement est réalisée \
                       à l'aide d'un modèle d'intelligence artificielle.\
                       Cet algorithme est entraîné sur les données de Home Credit.\
                       Ces donnnées sont constituées d'environ 300 000 demandes de prêt, \
                       pour lesquelles plusieurs centaines de caractéristiques \
                       relatives au client sont renseignées.\
                       Ce modèle, appelé Extreme Gradient Boosting, consiste\
                       à créer des arbres de décision aléatoires, puis à les\
                       améliorer étape par étape jusqu'à converger vers une \
                       précision de classification maximale.")
        
        
        st.header('__Interprétabilité de la prédiction de défaut de paiement__')
        
        #Selection du nombre de top variables à afficher pour le waterfall plot
        n_feats = st.slider(
                "Nombre de variables les plus influentes sur la prédiction du modèle",
                min_value=5,
                max_value=30,
                value=10,
                step=1,
                help="Sélectionnez entre 5 et 30 variables à afficher"
                )
        
        #Affichage du waterfall plot pour l'interprétabilité de la prédiction
        shap_series = -get_shap_values(select_id)
        fig = plotly_waterfall(select_id, shap_series, n_feats=n_feats)[0]
        st.plotly_chart(fig, use_container_width=True)
        
        
        #Explication du graphique
        wat_exp = st.beta_expander("Plus d'informations à propos de ce \
                                         graphique")
        wat_exp.write("_Ce graphique représente l'influence des variables\
                        sur la prédiction de probabilité de difficulté de\
                        paiement. En partant du bas du graphique:_")
        wat_exp.write("_- Au départ on se situe à 50, ce qui correspond à \
                        une probabilité de défaut de paiement de base de 50%._")
        wat_exp.write("_- Chaque barre correspond à l'influence d'une\
                        variable sur la capacité de remboursement du client.\
                        Si la barre est rouge, cela veut dire que \
                        la valeur de la variable est en sa \
                        défaveur. Si la barre est verte, cela veut dire que la\
                        valeur de la variable est en sa faveur. Plus la barre \
                        est longue, plus la variable est impactante._")
        wat_exp.write("_- En haut du graphique, la barre s'arrête au niveau de\
                         la probabilté de difficulté de paiement totale du \
                         client._")
        
        #Choix de la variable à afficher dans le graphique
        VARS = plotly_waterfall(select_id, shap_series, n_feats=n_feats)[1] 
        var = st.selectbox('Sélection de la variable à afficher', VARS)
        
        #Chargement des données
        data = load_data()
        
        if data[var].dtype == 'float64' and data[var].nunique() > 2:
            
            #Selection du nombre de top variables à afficher pour le waterfall plot
            n_bins = st.slider(
                    "Nombre de barres du graphique ci-dessous",
                    min_value=3,
                    max_value=30,
                    value=10,
                    step=1,
                    help="Sélectionnez entre 3 et 30 barres à afficher"
                    )
            
            #Affichage du graphique
            bars = plot_bars(data, var, select_id, n_bins=n_bins+1)
            st.plotly_chart(bars, use_container_width=True)
            
        else:
            #Affichage du graphique
            bars = plot_categ_bars(data, var, select_id)[0]
            st.plotly_chart(bars, use_container_width=True)
        
        
        #
        #Explication du graphique
        graph_exp = st.beta_expander("Plus d'informations à propos de ce \
                                         graphique")
        graph_exp.write("_Le client appartient à la classe correspondant à la barre colorée en violet.\
                        La hauteur d'une barrre est égale\
                        au pourcentage moyen de défaut de paiement des individus appartenant à cette classe.\
                        Ces statistiques descriptives ne sont pas directement utilisées par le modèle.\
                        L'affichage de ce graphique permet uniquement d'appuyer les raisons pour \
                        lesquelles le modèle attribue à cette variable un impact positif, ou \
                        négatif, sur la capacité de remboursment du client._")
       
        
        
    #--------------------------------------------------------------------------
    ### Informations relatives au client
    #--------------------------------------------------------------------------
    
    if radio == 'Informations relatives au client':
        st.header("__Affichage de quelques informations relatives au client__")
        st.subheader("Voici quelques statistiques.\
                     Pour chaque variable, il est possible d'afficher le __pourcentage \
                     de difficulté de paiement par groupe__, ainsi que __la distribution\
                      d'individus par groupe__ ")
        st.write('Sur chaque graphique, le client sera représenté en violet, \
                 et la moyenne du pourcentage global de défaut de paiement en pointillés orange')
        
        variables = load_data().columns
        
        
        var_list = var_list = ['Sexe', 'Jours travaillés', 'Ratio Revenu/Montant du prêt',
                    "Type d'éducation"]
        
        chosen_vars = st.multiselect('Choix des variables à modifier',
                                     variables,
                                     var_list)
        
        data = load_data()
        
        for i, var in enumerate(chosen_vars):
            st.subheader(f'Statistiques sur la variable {var}')
            left_col, right_col = st.beta_columns(2)
            
            if data[var].dtype == 'float64' and data[var].nunique() > 10:
            
                #Selection du nombre de top variables à afficher pour le waterfall plot
                n_bins = left_col.slider(
                        "Nombre de barres du graphique ci-dessous",
                        min_value=3,
                        max_value=30,
                        value=10,
                        step=1,
                        help="Sélectionnez entre 3 et 30 barres à afficher",
                        key= str(i)
                        )
                
                #Affichage du graphique taux de difficulté
                bars = plot_bars(data, var, select_id, n_bins=n_bins+1)
                left_col.plotly_chart(bars, use_container_width=True)
                
                #Selection du nombre de top variables à afficher pour le waterfall plot
                n_bins = right_col.slider(
                        "Nombre de barres du graphique ci-dessous",
                        min_value=3,
                        max_value=30,
                        value=10,
                        step=1,
                        help="Sélectionnez entre 3 et 30 barres à afficher",
                        key= 'dist'+str(i)
                        )
                #Affichage du graphique distribution 
                dist = plot_distribution(data, var, select_id, n_bins=n_bins)
                right_col.plotly_chart(dist, use_container_width=True)
            
            else:
                #Affichage du graphique taux de difficulté
                bars = plot_categ_bars(data, var, select_id)[0]
                left_col.plotly_chart(bars, use_container_width=True)
                
                #Affichage du graphique distribution 
                rep = plot_repartition(data, var, select_id)
                right_col.plotly_chart(rep, use_container_width=True)
                
            
    
    #--------------------------------------------------------------------------
    ### Modification des paramètres
    #--------------------------------------------------------------------------
    
    if radio == 'Modification des paramètres':
        st.header("__Et si on modifiait les valeurs de quelques variables ?__")
        st.subheader('Il est possible de mofifier certaines variables relatives au client du prêt sélectionné.\
                 Nous pourrons ainsi visualiser une nouvelle prédiction tenant compte de ces nouveau paramètres.')
        variables = load_data_full().columns
        var_list = ['Sexe', 'Jours travaillés', 'Ratio Revenu/Montant du prêt',
                    "Type d'éducation"]
        chosen_vars = st.multiselect('Choix des variables à modifier',
                                     variables,
                                     var_list)
        var_dict = {}
        for i, var in enumerate(chosen_vars):
            val = get_value_full(select_id, var)
            st.write(f'Valeur originale de {var}: {val}')
            
            if isinstance(val, float):
                var_dict[var] = st.slider(
                        f"Modifiez la valeur de {var}",
                        min_value=np.float(load_data_full()[var].min()),
                        max_value=np.float(load_data_full()[var].max()),
                        value=np.float(val)
                        )
                
            elif isinstance(val, np.integer):
                var_dict[var] = st.slider(
                        f"Modifiez la valeur de {var}",
                        min_value=int(load_data_full()[var].min()),
                        max_value=int(load_data_full()[var].max()),
                        value=int(val),
                        step=1
                        )
        

        left_col, right_col = st.beta_columns(2)
        
        #Prédiction sans modification
        left_col.header('Prédiction sans modification:')
        #Récpération de la probabilité de difficulté de paiement originale
        default_proba = int(100*(1-get_score(select_id)))
        left_fig = default_gauge(default_proba)
        #Jauge de probabilité de difficulté de paiement
        left_col.plotly_chart(left_fig, use_container_width=True)
        #Waterfall Plot sans mofications
        left_col.header('__Interprétabilité de la prédiction sans modification__')
        #Selection du nombre de top variables à afficher pour le waterfall plot
        n_feats_left = left_col.slider(
                "Nombre de variables les plus influentes sur la prédiction du modèle",
                min_value=5,
                max_value=30,
                value=10,
                step=1,
                help="Sélectionnez entre 5 et 30 variables à afficher"
                ) 
        #Affichage du waterfall plot pour l'interprétabilité de la prédiction
        shap_series = -get_shap_values(select_id)
        wat_fig_l = plotly_waterfall(select_id, shap_series, n_feats=n_feats_left)[0]
        left_col.plotly_chart(wat_fig_l, use_container_width=True)
        
        
        #Prédiction avec modifications
        right_col.header('Prédiction avec modifications:')
        with st.spinner('Prédiction en cours...'):
            def_proba_modif, X = predict_modified(load_data_full(), var_dict, select_id, path)
            right_fig = default_gauge(int(100*def_proba_modif))
            #Jauge de probabilité de difficulté de paiement
            right_col.plotly_chart(right_fig, use_container_width=True)
            #Waterfall Plot avec mofications
            right_col.header('__Interprétabilité de la prédiction avec modifications__')
            #Selection du nombre de top variables à afficher pour le waterfall plot
            n_feats_right = right_col.slider(
                    "Nombre de variables les plus influentes sur la prédiction du modèle",
                    min_value=5,
                    max_value=30,
                    value=10,
                    step=1,
                    help="Sélectionnez entre 5 et 30 variables à afficher",
                    key='left_slider'
                    ) 
            #Affichage du waterfall plot pour l'interprétabilité de la prédiction
            shap_series_bis = -get_shap_values(select_id)
            shap_modif = -transform_shap(def_proba_modif, X, path)
            
            wat_fig_r, var_list = plotly_waterfall(select_id,
                                         shap_series_bis,
                                         n_feats=n_feats_right,
                                         shap_modif=shap_modif
                                         )
            right_col.plotly_chart(wat_fig_r, use_container_width=True)
        

if __name__ == '__main__':
    main()