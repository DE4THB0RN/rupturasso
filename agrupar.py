import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import joblib

db = pd.read_csv('argentina_cars.csv')
db = db.drop(columns=['money','currency','brand'])
db = db.dropna()

label_encoder_modelo = LabelEncoder()
label_encoder_cor = LabelEncoder()
label_encoder_fuel = LabelEncoder()
label_encoder_gear = LabelEncoder()
label_encoder_motor = LabelEncoder()
label_encoder_body = LabelEncoder()

# Separar as colunas numéricas (conteúdo e cabeçalho)
db_numerico = db.select_dtypes(include=['number']).columns

# Separar as colunas categóricas (conteúdo e cabeçalho)
db_categorico = db.select_dtypes(include=['object', 'category']).columns


preprocessador = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), db_numerico),
        ('cat', OneHotEncoder(sparse_output=False), db_categorico)
    ]
)
#for indice in db_categorico['model']:
    # Mostrar o índice e o nome correspondente
#    nome_original = label_encoder.inverse_transform([indice])[0]
#    print(f"Índice {indice} -> Nome: {nome_original}")

db_full = preprocessador.fit_transform(db)



lowest_bic = np.infty
bic = []
n_components_range = range(1, 10)
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=42)
    gmm.fit(db_full)
    bic.append(gmm.bic(db_full))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

# O melhor número de componentes é escolhido com base no BIC
grupos = best_gmm.predict(db_full)

# Prever os grupos (clusters)
aic = best_gmm.aic(db_full)
bic = best_gmm.bic(db_full)
print(f"AIC: {aic}, BIC: {bic}")

# Avaliar o modelo usando o Silhouette Score (opcional)
silhouette_avg = silhouette_score(db_full, grupos)
print(f"Silhouette Score: {silhouette_avg}")

log_likelihood = best_gmm.score(db_full)  # score retorna o log-likelihood
print(f'Log-Likelihood: {log_likelihood}')

joblib.dump(preprocessador, 'preprocessador.pkl')
joblib.dump(best_gmm,'modelo_gmm.pkl')
