import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import argparse
import sys

def criar_boxplot(args):
    """
    Função principal para ler, filtrar e gerar o box plot a partir de um arquivo CSV.
    """
    # --- 1. Leitura do Arquivo CSV com Pandas ---
    try:
        # Tenta ler o CSV. 'sep' é o delimitador (ponto e vírgula no nosso caso).
        df = pd.read_csv(args.csv_path, sep=args.delimiter)
        print(f"Arquivo '{args.csv_path}' lido com sucesso. Total de {len(df)} linhas.")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{args.csv_path}' não foi encontrado.")
        sys.exit(1) # Encerra o script com código de erro
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo CSV: {e}")
        sys.exit(1)

    # --- 2. Lógica de Filtragem ---
    # Aplica o filtro se o usuário especificou uma coluna de filtro válida (>= 0)
    if args.filtro_col >= 0 and args.filtro_valor is not None:
        try:
            # Pega o nome da coluna de filtro a partir do seu índice
            nome_coluna_filtro = df.columns[args.filtro_col]
            print(f"Filtrando dados onde a coluna '{nome_coluna_filtro}' é igual a '{args.filtro_valor}'...")
            
            # O filtro poderoso do pandas: cria um novo DataFrame apenas com as linhas que correspondem ao critério.
            # Convertemos ambos para string para garantir uma comparação correta.
            df = df[df[nome_coluna_filtro].astype(str) == str(args.filtro_valor)].copy()
            
            if df.empty:
                print("Aviso: Nenhum dado encontrado após a aplicação do filtro.")
                sys.exit(0) # Encerra sem erro, pois a operação foi válida
                
            print(f"Filtragem resultou em {len(df)} linhas.")
        except IndexError:
            print(f"Erro: O índice da coluna de filtro '{args.filtro_col}' não existe no arquivo.")
            sys.exit(1)
            
    # --- 3. Preparação da Coluna de Dados (Datas) ---
    try:
        # Pega o nome da coluna de dados a partir do seu índice
        nome_coluna_dados = df.columns[args.col]
        print(f"Processando a coluna de dados '{nome_coluna_dados}'...")

        # Converte a coluna para o tipo datetime. Erros de conversão virarão 'NaT' (Not a Time).
        # O formato '%d/%m/%Y' corresponde a 'DD/MM/AAAA'.
        df[nome_coluna_dados] = pd.to_datetime(df[nome_coluna_dados], format='%d/%m/%Y', errors='coerce')

        # Remove as linhas onde a conversão da data falhou (valores NaT)
        linhas_antes = len(df)
        df.dropna(subset=[nome_coluna_dados], inplace=True)
        linhas_depois = len(df)
        
        if linhas_depois < linhas_antes:
            print(f"Aviso: {linhas_antes - linhas_depois} linhas foram removidas por terem datas em formato inválido.")

        if df.empty:
            print("Erro: Nenhum dado com data válida restou para gerar o gráfico.")
            sys.exit(1)

    except IndexError:
        print(f"Erro: O índice da coluna de dados '{args.col}' não existe no arquivo.")
        sys.exit(1)

    # --- 4. Criação e Salvamento do Gráfico ---
    print("Gerando o gráfico box plot...")
    plt.style.use('seaborn-v0_8-whitegrid') # Define um estilo visual agradável
    fig, ax = plt.subplots(figsize=(8, 10)) # Cria a figura e os eixos do gráfico

    # Seaborn simplifica a criação do box plot para uma única linha
    sns.boxplot(y=df[nome_coluna_dados], ax=ax, width=0.3)

    # Formatação do eixo Y para mostrar as datas corretamente
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))

    # Títulos e rótulos
    ax.set_title(f"Box Plot para '{nome_coluna_dados}'", fontsize=16)
    ax.set_ylabel("Data", fontsize=12)
    ax.set_xlabel(f"Filtro: {df.columns[args.filtro_col]} = {args.filtro_valor}" if args.filtro_col >= 0 else "Todos os Dados", fontsize=12)
    
    plt.tight_layout() # Ajusta o layout para evitar sobreposição

    try:
        plt.savefig(args.output_path)
        print(f"Gráfico salvo com sucesso em '{args.output_path}'")
    except Exception as e:
        print(f"Ocorreu um erro ao salvar o gráfico: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # --- Configuração dos Argumentos da Linha de Comando com argparse ---
    parser = argparse.ArgumentParser(description="Gera um gráfico box plot a partir de uma coluna de um arquivo CSV, com opção de filtro.")
    
    parser.add_argument("--csv", dest="csv_path", required=True, help="Caminho para o arquivo CSV de entrada.")
    parser.add_argument("--col", dest="col", type=int, required=True, help="Índice da coluna para gerar o box plot (base 0).")
    parser.add_argument("--filtro-col", dest="filtro_col", type=int, default=-1, help="Índice da coluna para usar como filtro. Desativado por padrão.")
    parser.add_argument("--filtro-valor", dest="filtro_valor", type=str, help="Valor a ser encontrado na coluna de filtro.")
    parser.add_argument("--delimiter", dest="delimiter", type=str, default=";", help="Delimitador de colunas do arquivo CSV.")
    parser.add_argument("--out", dest="output_path", type=str, default="boxplot_python.png", help="Nome do arquivo de imagem de saída.")

    args = parser.parse_args()
    
    criar_boxplot(args)