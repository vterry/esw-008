import pandas as pd
import numpy as np

def analisar_correlacao_covariancia(caminho_arquivo: str):
    """
    Realiza a análise de correlação e covariância da idade dos pacientes
    com outras variáveis numéricas em um arquivo CSV.

    Args:
        caminho_arquivo (str): O caminho para o arquivo CSV.
    """
    try:
        # 1. Carregar o arquivo CSV usando pandas.
        # O delimitador é definido como ';'.
        # O 'low_memory=False' ajuda a evitar avisos de tipo de dado misto em arquivos grandes.
        df = pd.read_csv(caminho_arquivo, delimiter=';', low_memory=False)

        # 2. Selecionar as colunas de interesse para a análise.
        # Estas são as mesmas colunas usadas no exemplo Go.
        colunas_analise = [
            'nu_idade_paciente',
            'co_dose_vacina',
            'co_vacina',
            'co_raca_cor_paciente',
            'co_local_aplicacao',
            'co_municipio_paciente',
            'co_cnes_estabelecimento',
        ]

        # Filtra o DataFrame para manter apenas as colunas de interesse.
        df_analise = df[colunas_analise].copy()

        # 3. Limpeza dos Dados: Converter para numérico e remover linhas com dados faltantes.
        # O 'pd.to_numeric' converte os dados, e 'coerce' transforma erros em 'NaN' (Not a Number).
        for col in colunas_analise:
            df_analise[col] = pd.to_numeric(df_analise[col], errors='coerce')

        # 'dropna()' remove qualquer linha que contenha pelo menos um valor NaN.
        df_analise.dropna(inplace=True)

        print(f"Análise baseada em {len(df_analise)} registros válidos.\n")

        # 4. Isolar a variável principal para a análise.
        idade_data = df_analise['nu_idade_paciente']

        # 5. Calcular e exibir a Correlação e Covariância.
        print("--- Análise de Correlação e Covariância com 'nu_idade_paciente' ---")
        print(f"{'Variável':<25} | {'Correlação':<15} | {'Covariância':<15}")
        print("-" * 60)

        for coluna in df_analise.columns:
            if coluna == 'nu_idade_paciente':
                continue

            outra_coluna_data = df_analise[coluna]

            # O método .corr() do pandas calcula a Correlação de Pearson.
            correlacao = idade_data.corr(outra_coluna_data)

            # O método .cov() do pandas calcula a Covariância.
            covariancia = idade_data.cov(outra_coluna_data)

            print(f"{coluna:<25} | {correlacao:<15.4f} | {covariancia:<15.4f}")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

# --- Execução da análise ---
# Certifique-se de que o arquivo 'vacinas-sp-jan-2025.csv' está no mesmo diretório.
caminho_do_arquivo = '/content/vacinas-sp-jan-2025.csv'
analisar_correlacao_covariancia(caminho_do_arquivo)