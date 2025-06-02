import pandas as pd

# 1. Leitura do CSV
df = pd.read_csv('Graficos_FakeRecogna.csv')

# 2. Converter coluna "Data" para datetime
df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

# 3. Transformar "Data" em número — timestamp ou ordinal
df['Data_num'] = df['Data'].map(pd.Timestamp.toordinal)  # transforma para número de dias

# 4. Converter "Classe" para numérico, se não for
df['Classe_num'] = pd.to_numeric(df['Classe'], errors='coerce')

# 5. Remover linhas com NaN nas colunas usadas
df_corr = df[['Data_num', 'Classe_num']].dropna()

# 6. Calcular correlação
correlacao = df_corr['Data_num'].corr(df_corr['Classe_num'])

# 7. Calcular covariância
covariancia = df_corr['Data_num'].cov(df_corr['Classe_num'])

# 8. Mostrar resultados
print(f'Correlação: {correlacao}')
print(f'Covariância: {covariancia}')
