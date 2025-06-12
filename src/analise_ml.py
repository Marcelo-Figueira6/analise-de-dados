# -*- coding: utf-8 -*-
"""
Script Python para Análise de Dados e Machine Learning Básico.

Este script demonstra um fluxo completo, desde o carregamento de dados
até a avaliação de um modelo simples de Machine Learning, utilizando
Pandas, NumPy, Matplotlib, Seaborn e Scikit-learn.

Ideal para ser executado em células de um Jupyter Notebook.
"""

# %% [markdown]
# # Análise de Dados e Machine Learning com Python
#
# Este notebook demonstra um fluxo de trabalho básico para análise exploratória de dados (EDA),
# pré-processamento e aplicação de um modelo de Machine Learning (Regressão Linear).

# %% [code]
# ==============================================================================
# 0. Importação das Bibliotecas Necessárias
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Preparando as ferramentas necessárias para a análise de dados...")

# %% [markdown]
# # 1. Carregamento de Dados
#
# Carreguei o dataset a partir de um arquivo CSV.

# %% [code]
# ==============================================================================
# 1. Função para Carregar Dados
# ==============================================================================
def carregar_dados(caminho_arquivo):
    """Carrega dados de um arquivo CSV para um DataFrame Pandas.

    Args:
        caminho_arquivo (str): O caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame carregado ou None se ocorrer erro.
    """
    try:
        df = pd.read_csv(caminho_arquivo)
        print(f"\nDados carregados com sucesso! foi encontrado {df.shape[0]} registros e {df.shape[1]} tipos de informação.")
        return df
    except FileNotFoundError:
        print(f"\nOops! Não consegui encontrar o arquivo em: {caminho_arquivo}")
        print("Por favor, verifique se o arquivo está no local correto.")
        return None
    except Exception as e:
        print(f"\nOcorreu um problema ao carregar o arquivo: {e}")
        print("Pode ser um erro no formato do arquivo ou no conteúdo.")
        return None

# %% [markdown]
# # 2. Análise Exploratória de Dados (EDA)
#
# Realizei uma análise inicial para entender a estrutura e as características dos dados.

# %% [code]
# ==============================================================================
# 2. Funções para Análise Exploratória de Dados (EDA)
# ==============================================================================
def exibir_inicio_fim(df, n=5):
    """Exibe as primeiras e últimas n linhas do DataFrame."""
    if df is not None:
        print("\n--- Dando uma olhada nos primeiros e últimos registros ---")
        print(df.head(n).to_string()) # Usar to_string() para garantir que tudo seja exibido
        print("\n--- ... ---")
        print(df.tail(n).to_string()) # Usar to_string()
    else:
        print("Não foi possível ver os registros, pois os dados não foram carregados.")

def exibir_info_gerais(df):
    """Exibe informações gerais sobre o DataFrame (tipos, não nulos)."""
    if df is not None:
        print("\n--- Entendendo os tipos de informações e se há dados faltando ---")
        df.info(verbose=True)
        print("\nIsso nos mostra quantas entradas temos para cada tipo de informação (coluna) e o tipo de dado (texto, número, etc.).")
    else:
        print("Não foi possível obter informações gerais, pois os dados não foram carregados.")

def exibir_estatisticas_descritivas(df):
    """Exibe estatísticas descritivas para colunas numéricas."""
    if df is not None:
        print("\n--- Resumo das informações numéricas (média, mínimo, máximo, etc.) ---")
        df_numeric = df.select_dtypes(include=np.number)
        if not df_numeric.empty:
            print(df_numeric.describe().to_string()) # Usar to_string()
            print("\nEste é um resumo rápido dos números em suas colunas.")
        else:
            print("Não foi encontrada nenhuma coluna com números para resumir.")
    else:
        print("Não foi possível obter estatísticas, pois os dados não foram carregados.")

def verificar_valores_ausentes(df):
    """Verifica e exibe a contagem de valores ausentes por coluna."""
    if df is not None:
        print("\n--- Verificando se há dados faltando ---")
        ausentes = df.isnull().sum()
        ausentes = ausentes[ausentes > 0]
        if not ausentes.empty:
            print("Sim, foi encontrado alguns dados faltando nas seguintes colunas:")
            print(ausentes.to_string()) # Usar to_string()
            print(f"\nNo total, faltam {ausentes.sum()} informações.")
        else:
            print("Ótimo! Não foi encontrada nenhuma informação faltando (dados ausentes).")
    else:
        print("Não foi possível verificar dados faltando, pois os dados não foram carregados.")

def verificar_valores_unicos(df, colunas_categoricas):
    """Exibe a contagem de valores únicos para colunas categóricas especificadas."""
    if df is not None:
        print("\n--- Contando as opções em categorias de texto selecionadas ---")
        for coluna in colunas_categoricas:
            if coluna in df.columns:
                print(f"\nOpções na coluna: '{coluna}'")
                # Mostra também a quantidade de NaNs se houver, tratando-os como 'Não Informado'
                print(df[coluna].value_counts(dropna=False).rename('Contagem').to_string()) # Usar to_string()
                print("Isso nos mostra quais opções aparecem e com que frequência.")
            else:
                print(f"Atenção: A coluna '{coluna}' que você pediu para verificar não foi encontrada.")
    else:
        print("Não foi possível verificar as opções em categorias, pois os dados não foram carregados.")

# %% [markdown]
# # 3. Pré-processamento de Dados
#
# Preparei os dados para a modelagem, tratando valores ausentes e codificando variáveis categóricas.

# %% [code]
# ==============================================================================
# 3. Funções para Pré-processamento
# ==============================================================================
def tratar_valores_ausentes(df, estrategia_num= 'median', estrategia_cat='most_frequent'):
    """Trata valores ausentes usando SimpleImputer. """
    if df is None:
        print("Não foi possível preencher os dados faltando, pois os dados não foram carregados.")
        return None

    df_tratado = df.copy()
    colunas_com_ausentes = df_tratado.columns[df_tratado.isnull().any()].tolist()

    if not colunas_com_ausentes:
        print("Não há informações faltando para preencher.")
        return df_tratado

    print(f"\n--- Preenchendo as informações que estão faltando em {len(colunas_com_ausentes)} colunas ---")

    colunas_num = df_tratado[colunas_com_ausentes].select_dtypes(include=np.number).columns
    colunas_cat = df_tratado[colunas_com_ausentes].select_dtypes(exclude=np.number).columns

    if not colunas_num.empty:
        imputer_num = SimpleImputer(strategy=estrategia_num)
        df_tratado[colunas_num] = imputer_num.fit_transform(df_tratado[colunas_num])
        print(f"Informações numéricas faltando ({', '.join(colunas_num)}) foram preenchidas usando a {estrategia_num} dos valores existentes.")

    if not colunas_cat.empty:
        imputer_cat = SimpleImputer(strategy=estrategia_cat)
        df_tratado[colunas_cat] = imputer_cat.fit_transform(df_tratado[colunas_cat])
        print(f"Informações de texto faltando ({', '.join(colunas_cat)}) foram preenchidas com a opção mais comum.")

    print("Preenchimento de informações faltando concluído.")
    verificar_valores_ausentes(df_tratado) 
    return df_tratado

def codificar_variaveis_categoricas(df, colunas_categoricas, metodo= 'onehot'):
    if df is None:
        print("Não foi possível transformar as categorias, pois os dados não foram carregados.")
        return None
    df_codificado = df.copy()
    print(f"\n--- Transformando categorias de texto em números (Método: {metodo}) ---")

    colunas_existentes = [col for col in colunas_categoricas if col in df_codificado.columns]
    if not colunas_existentes:
        print("Não foi encontrada as colunas de categorias que você pediu para transformar.")
        return df_codificado

    if metodo =='label':
        encoder = LabelEncoder()
        for coluna in colunas_existentes:
            if df_codificado[coluna].isnull().any():
                df_codificado[coluna].fillna( 'Desconhecido', inplace=True) 
            df_codificado[coluna] = encoder.fit_transform(df_codificado[coluna].astype(str))
            print(f"Coluna  '{coluna}' transformada para números de 0 a X.")
    elif metodo =='onehot':
        df_codificado = pd.get_dummies(df_codificado, columns=colunas_existentes, drop_first=True, dummy_na=False)
        print(f"Colunas {colunas_existentes} foram transformadas em várias novas colunas com 0s e 1s.")
    else:
        print(f"Desculpe, o método de transformação '{metodo}' não é reconhecido. Use 'label' ou 'onehot'.")
        return df

    print("Transformação de categorias concluída.")
    return df_codificado

def remover_colunas_nao_numericas_para_modelo(df, coluna_target):

    if df is None:
        return None

    df_limpo = df.copy()
    colunas_para_remover = []

    for col in df_limpo.columns:
        if col != coluna_target and not pd.api.types.is_numeric_dtype(df_limpo[col]):
            colunas_para_remover.append(col)

    if colunas_para_remover:
        print(f"\n--- Removendo colunas de texto que não servem para o modelo: {', '.join(colunas_para_remover)} ---")
        df_limpo = df_limpo.drop(columns=colunas_para_remover)
        print("Colunas de texto indesejadas removidas antes de treinar o modelo.")
    else:
        print("\nTodas as colunas, exceto a alvo, já são numéricas ou foram transformadas. Ótimo!")

    return df_limpo

def dividir_dados_treino_teste(df, coluna_target, test_size=0.3, random_state=42):
    """Divide o DataFrame em conjuntos de treino e teste."""
    if df is None or coluna_target not in df.columns:
        print(f"Erro: Não foi possível dividir os dados. A coluna principal '{coluna_target}' não foi encontrada ou os dados estão vazios.")
        return None, None, None, None

    print("\n--- Separando os dados para 'aprender' e para 'testar' ---")
    X = df.drop(coluna_target, axis=1) # O que o modelo vai aprender
    y = df[coluna_target] # O que o modelo vai prever

    if X.empty or y.empty:
        print("Erro: Os dados para aprender (X) ou o que deve ser previsto (y) ficaram vazios após a separação.")
        return None, None, None, None

    try:
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Dados separados: {X_treino.shape[0]} amostras para o modelo aprender, e {X_teste.shape[0]} amostras para testá-lo.")
        return X_treino, X_teste, y_treino, y_teste
    except Exception as e:
        print(f"Ocorreu um problema ao separar os dados para treino e teste: {e}")
        return None, None, None, None

# %% [markdown]
# # 4. Visualização de Dados
#
# Criei gráficos para visualizar a distribuição e as relações nos dados.

# %% [code]
# ==============================================================================
# 4. Funções para Visualização de Dados (com comentários narrativos)
# ==============================================================================
def plotar_histograma(df, coluna_numerica, bins=10):
    """Plota um histograma para visualizar a distribuição de uma coluna numérica.

    O histograma agrupa os valores da coluna em 'bins' (intervalos) e mostra,
    através da altura das barras, quantos valores caem em cada intervalo.
    A curva KDE (Kernel Density Estimate) suavizada ajuda a visualizar a forma
    geral da distribuição de probabilidade dos dados.
    """
    if df is not None and coluna_numerica in df.columns and pd.api.types.is_numeric_dtype(df[coluna_numerica]):
        print(f"\n--- Criando um gráfico de distribuição para '{coluna_numerica}' ---")
        plt.figure(figsize=(8, 5))

        sns.histplot(df[coluna_numerica].dropna(), kde=True, bins=bins)

        plt.title(f'Como os valores de {coluna_numerica} se distribuem')
        plt.xlabel(coluna_numerica)
        plt.ylabel('Quantas vezes aparecem')
        plt.grid(axis='y', alpha=0.5)
        plt.show()

    elif df is None:
        print("Não foi possível criar o histograma, pois os dados não foram carregados.")
    elif coluna_numerica not in df.columns:
        print(f"Erro: Não foi encontrada a coluna '{coluna_numerica}' para o histograma.")
    else:
        print(f"Erro: A coluna '{coluna_numerica}' não tem números para criar um histograma.")

def plotar_grafico_barras(df, coluna_categorica):
    """Plota um gráfico de barras para visualizar a frequência de cada categoria
    em uma coluna categórica.

    Cada barra representa uma categoria única presente na coluna. A altura da
    barra indica quantas vezes essa categoria aparece no dataset (sua contagem).
    As barras são ordenadas da mais frequente para a menos frequente.
    """
    if df is not None and coluna_categorica in df.columns:
        # --- ALTERAÇÃO AQUI: Atualiza a verificação de tipo de dado ---
        # A forma mais moderna e recomendada pelo Pandas para verificar se a coluna é categórica ou de texto
        if isinstance(df[coluna_categorica].dtype, pd.CategoricalDtype) or df[coluna_categorica].dtype == 'object':
            print(f"\n--- Criando um gráfico de barras para a coluna '{coluna_categorica}' ---")
            plt.figure(figsize=(8, 5))

            # Cria o gráfico de barras usando Seaborn.
            # 'x=coluna_categorica' define a variável categórica para o eixo X.
            # 'data=df.fillna({coluna_categorica: 'Não Informado'})' usa o DataFrame, tratando NaNs como uma categoria 'Não Informado'.
            # 'order=...' ordena as barras pela frequência (do maior para o menor).
            sns.countplot(x=coluna_categorica, data=df.fillna({coluna_categorica: 'Não Informado'}), order=df[coluna_categorica].value_counts(dropna=False).index)

            # --- ALTERAÇÃO AQUI: Melhoria no título do gráfico ---
            plt.title(f'Quantas vezes cada tipo de "{coluna_categorica}" aparece')

            # Define o rótulo do eixo X, que representa as diferentes categorias.
            plt.xlabel(coluna_categorica)

            # Define o rótulo do eixo Y, que representa a contagem (frequência) de cada categoria.
            plt.ylabel('Contagem')

            # Rotaciona os rótulos do eixo X para melhor legibilidade se houver muitas categorias.
            plt.xticks(rotation=45, ha='right')

            # Adiciona uma grade horizontal.
            plt.grid(axis='y', alpha=0.5)

            # Ajusta o layout para evitar sobreposição de elementos.
            plt.tight_layout()

            # Exibe o gráfico.
            plt.show()
        else:
            print(f"Erro: A coluna '{coluna_categorica}' não parece ser uma categoria (texto).")
    elif df is None:
        print("Não foi possível criar o gráfico de barras, pois os dados não foram carregados.")
    else:
        print(f"Erro: Não encontramos a coluna '{coluna_categorica}' para o gráfico de barras.")

def plotar_grafico_dispersao(df, coluna_x, coluna_y):
    """Plota um gráfico de dispersão (scatter plot) para visualizar a relação
    entre duas colunas numéricas.

    Cada ponto no gráfico representa uma linha (observação) do DataFrame.
    A posição horizontal do ponto é determinada pelo valor da 'coluna_x'.
    A posição vertical do ponto é determinada pelo valor da 'coluna_y'.
    Este gráfico ajuda a identificar padrões como correlação (linear ou não),
    clusters (agrupamentos) ou outliers (pontos distantes).
    """
    if df is not None and coluna_x in df.columns and coluna_y in df.columns:
        if pd.api.types.is_numeric_dtype(df[coluna_x]) and pd.api.types.is_numeric_dtype(df[coluna_y]):
            print(f"\n--- Criando um gráfico para ver a relação entre '{coluna_x}' e '{coluna_y}' ---")
            plt.figure(figsize=(8, 5))

            sns.scatterplot(x=coluna_x, y=coluna_y, data=df)

            plt.title(f'Relação entre {coluna_x} e {coluna_y}')
            plt.xlabel(coluna_x)
            plt.ylabel(coluna_y)
            plt.grid(True, alpha=0.5)
            plt.show()
        else:
            print(f"Erro: Pelo menos uma das colunas ('{coluna_x}', '{coluna_y}') não tem números para criar este gráfico.")
    elif df is None:
        print("Não foi possível criar o gráfico de relação, pois os dados não foram carregados.")
    else:
        print(f"Erro: Não foi encontrado uma ou ambas as colunas ('{coluna_x}', '{coluna_y}') para o gráfico de relação.")

# %% [markdown]
# # 5. Modelagem de Machine Learning (Regressão Linear)
#
# Treinei um modelo de Regressão Linear e avaliamos seu desempenho.

# %% [code]
# ==============================================================================
# 5. Funções para Modelagem e Avaliação
# ==============================================================================
def treinar_modelo_regressao(X_treino, y_treino):
    """Treina um modelo de Regressão Linear."""
    if X_treino is None or y_treino is None:
        print("Erro: Não há dados válidos para o modelo aprender.")
        return None
    if X_treino.empty or y_treino.empty:
        print("Erro: Os dados que o modelo precisa para aprender estão vazios.")
        return None

    print("\n--- Ensinando o modelo a fazer previsões (Regressão Linear) ---")
    try:
        modelo = LinearRegression()
        modelo.fit(X_treino, y_treino)
        print("O modelo aprendeu com os dados com sucesso!")
        return modelo
    except Exception as e:
        print(f"Ops! Ocorreu um problema enquanto o modelo tentava aprender. Detalhes: {e}")
        print("Isso pode acontecer se houver dados inesperados ou formatos incorretos.")
        return None

def avaliar_modelo(modelo, X_teste, y_teste):
    """Avalia o modelo treinado usando MAE e R²."""
    if modelo is None or X_teste is None or y_teste is None:
        print("Erro: Não há modelo treinado ou dados de teste para avaliar.")
        return None, None
    if X_teste.empty or y_teste.empty:
        print("Erro: Os dados para testar o modelo estão vazios.")
        return None, None

    print("\n--- Verificando o quão bem o modelo prevê resultados novos ---")
    try:
        y_pred = modelo.predict(X_teste)

        mae = mean_absolute_error(y_teste, y_pred)
        r2 = r2_score(y_teste, y_pred)

        # Mensagens mais amigáveis
        print(f"A diferença média entre o que o modelo previu e o valor real foi de: {mae:.2f}")
        print(f"O modelo conseguiu explicar {r2*100:.2f}% da variação nos dados. Quanto mais perto de 100%, melhor!")

        plt.figure(figsize=(8, 5))
        plt.scatter(y_teste, y_pred, alpha=0.6)
        plt.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], '--r', linewidth=2)
        plt.xlabel("Valores Reais (o que realmente aconteceu)")
        plt.ylabel("Valores Previstos (o que o modelo estimou)")
        plt.title("Comparação: Valores Reais vs. Valores Previstos pelo Modelo")
        plt.grid(True, alpha=0.3)
        plt.show()

        return mae, r2
    except Exception as e:
        print(f"Ocorreu um problema ao avaliar o modelo. Detalhes: {e}")
        return None, None

# %% [markdown]
# # 6. Execução do Fluxo Completo (Célula a Célula)
#
# **IMPORTANTE:** Para executar este fluxo em um ambiente de notebook (como Jupyter ou VS Code),
# **certifique-se de executar TODAS as células anteriores** (importações e definições de funções - Seções 0 a 5)
# antes de executar as células abaixo (Seção 6). Se você reiniciar o kernel, precisará reexecutar
# as células de definição novamente.

# %% [code]
# ==============================================================================
# 6.1 Definições e Carregamento de Dados
# ==============================================================================
# Definições
# Ajuste o caminho se executar de um local diferente da raiz do projeto
# Ex: Se executar de /notebooks, use '../data/raw/dados_exemplo.csv'
# Ex: Se executar da raiz do projeto: caminho_csv = 'data/raw/dados_exemplo.csv'
caminho_csv = '../data/raw/dados_exemplo_2.csv' # Caminho relativo à pasta 'src'

# Identifique aqui todas as colunas que são de TEXTO/CATEGORIA no seu arquivo original
# Elas serão usadas para a Análise Exploratória e para a Codificação
colunas_categoricas_para_eda = ['sexo', 'categoria', 'observacao'] # Adicionei 'observacao'
colunas_categoricas_para_codificar = ['sexo', 'categoria'] 
# Para este código, a melhor opção é removê-la se não for codificada.

coluna_target = 'target'
coluna_numerica_hist = 'idade'
coluna_categorica_bar = 'categoria'
coluna_dispersao_x = 'valor_compra'
coluna_dispersao_y = 'target'

# --- Carregar Dados ---
import os
if os.path.basename(os.getcwd()) == 'src':
    print("Ajustando caminho para execução a partir da pasta 'src'...")
    pass
elif os.path.basename(os.getcwd()) == 'analise-de-dados':
    print("Executando da raiz do projeto. Ajustando caminho do CSV...")
    caminho_csv = 'data/raw/dados_exemplo.csv'
else:
    print(f"Atenção: Você está executando de: {os.getcwd()}. Verifique se o 'caminho_csv' está correto para o seu ambiente.")

df = carregar_dados(caminho_csv)

# %% [code]
# ==============================================================================
# 6.2 Análise Exploratória de Dados (EDA)
# ==============================================================================
if df is not None:
    exibir_inicio_fim(df)
    exibir_info_gerais(df)
    exibir_estatisticas_descritivas(df)
    verificar_valores_ausentes(df)
    # Use a lista atualizada de colunas categóricas para EDA
    verificar_valores_unicos(df, colunas_categoricas_para_eda)
else:
    print("Não foi possível realizar a Análise Exploratória de Dados, pois os dados não foram carregados.")

# %% [code]
# ==============================================================================
# 6.3 Visualização Inicial
# ==============================================================================
if df is not None:
    plotar_histograma(df, coluna_numerica_hist)
    plotar_grafico_barras(df, coluna_categorica_bar)
    plotar_grafico_dispersao(df, coluna_dispersao_x, coluna_dispersao_y)
else:
    print("Não foi possível criar os gráficos, pois os dados não foram carregados.")

# %% [code]
# ==============================================================================
# 6.4 Pré-processamento: Remoção de ID e Tratamento de Ausentes
# ==============================================================================
df_tratado = None # Initialize
if df is not None:
    # Remover coluna ID antes de processar/modelar, se não for útil
    if 'id' in df.columns:
        df_processado = df.drop('id', axis=1).copy()
        print("\nA coluna 'id' (identificação) foi removida, pois não é útil para o modelo.")
    else:
        df_processado = df.copy()

    df_tratado = tratar_valores_ausentes(df_processado, estrategia_num='median', estrategia_cat='most_frequent')
else:
    print("Não foi possível preparar os dados, pois houve um problema no carregamento inicial.")


# %% [code]
# ==============================================================================
# 6.5 Pré-processamento: Codificação de Variáveis Categóricas e Limpeza Final
# ==============================================================================
df_codificado = None # Initialize
if df_tratado is not None:
    # Certifique-se de que as colunas a codificar ainda existem após o tratamento
    colunas_cat_existentes = [col for col in colunas_categoricas_para_codificar if col in df_tratado.columns]
    df_codificado = codificar_variaveis_categoricas(df_tratado, colunas_cat_existentes, metodo='onehot')

    # NOVA ETAPA: Remover colunas que ainda são de texto e não foram codificadas
    if df_codificado is not None:
        df_final_para_modelo = remover_colunas_nao_numericas_para_modelo(df_codificado, coluna_target)
    else:
        df_final_para_modelo = None

else:
    print("Não foi possível transformar as categorias, pois houve um problema no tratamento de informações faltando.")


# %% [code]
# ==============================================================================
# 6.6 Divisão Treino/Teste
# ==============================================================================
X_treino, X_teste, y_treino, y_teste = None, None, None, None # Initialize
if df_final_para_modelo is not None: # Use o DataFrame final limpo
    X_treino, X_teste, y_treino, y_teste = dividir_dados_treino_teste(df_final_para_modelo, coluna_target)
else:
    print("\nNão foi possível separar os dados para o modelo devido a problemas nas etapas anteriores de preparação.")

# %% [code]
# ==============================================================================
# 6.7 Modelagem e Avaliação
# ==============================================================================
if X_treino is not None and y_treino is not None: # Verifica se a divisão foi bem-sucedida
    modelo_final = treinar_modelo_regressao(X_treino, y_treino)
    if modelo_final and X_teste is not None and y_teste is not None:
        avaliar_modelo(modelo_final, X_teste, y_teste)
    elif not modelo_final:
          print("\nO modelo não foi treinado. Verifique as mensagens de erro acima.")
    else:
          print("\nOs dados para testar o modelo não estão disponíveis.")

else:
    print("\nNão foi possível continuar com o treinamento e avaliação do modelo devido a problemas na separação dos dados.")

# %% [code]
print("\n--- Fim da Análise de Dados e Construção do Modelo ---")
print("O processo de aprendizado do modelo foi concluído. Os gráficos e as mensagens acima mostram os resultados.")