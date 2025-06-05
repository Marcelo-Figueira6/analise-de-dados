# Projeto de Análise de Dados e Machine Learning

Este projeto demonstra um fluxo básico de análise exploratória de dados (EDA), pré-processamento e aplicação de um modelo de Machine Learning (Regressão Linear) utilizando Python.

## Estrutura do Projeto

```
analise-de-dados/
├── .venv/                   # Ambiente virtual (ignorado pelo Git)
├── data/
│   ├── raw/                 # Dados brutos (ex: dados_exemplo.csv)
│   └── processed/           # Dados processados (se houver)
├── notebooks/               # Jupyter Notebooks (se usar)
├── src/
│   ├── __init__.py
│   └── analise_ml.py        # Script principal com funções de análise e ML
│   └── (seus módulos .py aqui)
├── models/                  # Modelos de ML treinados (se salvar)
├── reports/                 # Gráficos, relatórios gerados
├── .gitignore               # Arquivos a serem ignorados pelo Git
├── README.md                # Este arquivo
├── requirements.txt         # Dependências do projeto
└── main.py                  # (Opcional, script principal para orquestração)
```

## Como Usar

Para utilizar este projeto, siga os passos abaixo:

### 1. Preparação do Ambiente

Você deve configurar um ambiente virtual para isolar as dependências do projeto.

1.  **Clone o repositório** (se aplicável) ou descompacte o arquivo ZIP em uma pasta de sua escolha.
    ```bash
    git clone [https://github.com/SeuUsuario/NomeDoSeuRepositorio.git](https://github.com/SeuUsuario/NomeDoSeuRepositorio.git)
    cd NomeDoSeuRepositorio # Substitua pelo nome real da pasta do seu repositório
    ```
    
2.  **Crie e ative o ambiente virtual:**
    * **Crie o ambiente virtual:**
        ```bash
        python -m venv .venv
        ```
    * **Ative o ambiente virtual:**
        * **No Windows (Prompt de Comando):**
            ```bash
            .venv\Scripts\activate.bat
            ```
        * **No macOS/Linux (Terminal):**
            ```bash
            source .venv/bin/activate
            ```
        O ambiente estará ativo quando `(.venv)` aparecer no início da linha de comando.

3.  **Instale as dependências:**
    Com o ambiente virtual ativo, instale todas as bibliotecas necessárias listadas no arquivo `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### 2. Configuração do VS Code:

1.  **Abra o projeto no VS Code:**
    No terminal (com o ambiente virtual ativo e na pasta raiz do projeto):
    ```bash
    code .
    ```

2.  **Selecione o Intérprete Python:**
    * No VS Code, abra a Paleta de Comandos (`Ctrl+Shift+P` ou `Cmd+Shift+P`).
    * Digite `Python: Select Interpreter` e selecione-o.
    * Escolha o seu ambiente virtual `(.venv)`.
    * Verifique no canto inferior esquerdo do VS Code se o intérprete selecionado é `(.venv)`.

### 3. Preparação dos Dados

O projeto requer um arquivo de dados para análise.

1.  **Certifique-se de que um arquivo `.csv` esteja presente** na pasta `data/raw/`.
    * Utilize o arquivo de exemplo fornecido neste repositório (dados_exemplo.csv ou o 2).

### 4. Execução do Código

Duas opções principais estão disponíveis para executar o código:

#### Opção A: Executar como Jupyter Notebook (Recomendado para Análise Interativa)

Esta abordagem permite explorar a análise passo a passo e visualizar os gráficos diretamente no VS Code.

1.  **Crie um novo Jupyter Notebook:**
    * No VS Code, no painel "Explorer", clique com o botão direito na pasta `notebooks`.
    * Selecione `New File...` e nomeie-o (ex: `01_analise_interativa.ipynb`).
2.  **Copie o conteúdo de `src/analise_ml.py` para o Notebook:**
    * Abra `src/analise_ml.py` no VS Code.
    * Copie todo o conteúdo (`Ctrl+A`, `Ctrl+C`).
    * Cole na primeira célula do novo notebook. O VS Code dividirá automaticamente o código em células.
3.  **Selecione o Kernel do Notebook:**
    * No canto superior direito do notebook, clique onde está o kernel e selecione seu ambiente `(.venv)`.
4.  **Ajuste o Caminho do Arquivo CSV (se necessário):**
    * Na última célula de código (`# 6. Execução Principal`), confirme que o `caminho_csv` está definido como:
        ```python
        caminho_csv = '../data/raw/dados_exemplo.csv'
        ```

