# DemoML - Pipeline de Machine Learning para Manutenção Prescritiva

Sistema de Machine Learning para **manutenção prescritiva** em equipamentos industriais de produção.

## Visão Geral

Este pipeline implementa um fluxo completo de ciência de dados para manutenção prescritiva, desde a coleta de dados até
a geração de relatórios PDF com **previsões baseadas em ML**. O objetivo é prever quantos dias faltam até a próxima
manutenção de cada equipamento com base em:

- Dados históricos de produção
- **Medições de desgaste** (cilindro e fuso)
- **Estado atual do equipamento** (produção acumulada, taxa de refugo, índice de desgaste)

**Melhor Modelo:** XGBoost

**Equipamentos Monitorados:** 10 equipamentos demo (EQ-101 a EQ-110)

## Requisitos

### Sistema
- Python 3.8+
- Sistema operacional: Linux/Windows/macOS

### Dependências Python

```bash
# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Instalar dependências
pip install -r requirements.txt
```

## Estrutura do Projeto

```
ml_pipeline_demo/
├── scripts/                     # Scripts do pipeline
│   ├── run_pipeline.py          # Orquestrador principal
│   ├── auto_pipeline.py         # Automação com detecção de alterações
│   ├── s01_data_collection.py   # Etapa 1: Coleta e integração
│   ├── s02_preprocessing.py     # Etapa 2: Pré-processamento
│   ├── s03_eda.py               # Etapa 3: Análise exploratória
│   ├── s03b_advanced_eda.py     # Etapa 3b: EDA avançado
│   ├── s04_modeling.py          # Etapa 4: Modelagem
│   ├── s05_evaluation.py        # Etapa 5: Avaliação
│   ├── s06_generate_report.py   # Etapa 6: Relatório PDF
│   └── history_manager.py       # Gerenciador de histórico
│
├── config/
│   └── paths.py                 # Configuração centralizada de caminhos
│
├── data/
│   ├── manutenção/              # Dados de manutenção
│   │   └── dados_manutenção.csv # Histórico de manutenção
│   └── raw/                     # Dados brutos de produção
│       └── EQ-*.csv             # Arquivos por equipamento
│
├── outputs/                     # Saídas do pipeline
│   ├── eda_plots/               # Gráficos gerados
│   ├── models/                  # Modelos treinados
│   ├── history/                 # Histórico de execuções
│   └── Report_DemoML_RX.pdf     # Relatório final
│
├── generate_dummy_data.py       # Script para gerar dados dummy
├── requirements.txt             # Dependências Python
└── README.md
```

## Execução Rápida

### 1. Gerar Dados Dummy (opcional - já incluídos)

```bash
python generate_dummy_data.py
```

### 2. Executar Pipeline Completo

```bash
cd scripts
python run_pipeline.py
```

O pipeline executará todas as etapas automaticamente e gerará o relatório PDF.

### 3. Execução Automática

```bash
# Verificar alterações e executar se necessário
python auto_pipeline.py

# Monitoramento contínuo (verifica a cada 5 minutos)
python auto_pipeline.py --watch

# Forçar reexecução
python auto_pipeline.py --force
```

### 4. Verificar Resultados

- Relatório PDF: `outputs/Report_DemoML_RX.pdf`
- Métricas: `outputs/evaluation_report.txt`
- Gráficos: `outputs/eda_plots/`

## Fluxo do Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  ETAPA 1: COLETA E INTEGRAÇÃO                                   │
│  CSV Files (EQ-101, EQ-102, ...) → DataFrame Único              │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 2: PRÉ-PROCESSAMENTO E LIMPEZA                           │
│  - Remoção de duplicatas, tratamento de nulos                   │
│  - Cálculo de dias até manutenção (variável target)             │
│  - Features de medição/desgaste                                 │
│  - One-Hot Encoding                                             │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 3: ANÁLISE EXPLORATÓRIA (EDA)                            │
│  - Estatísticas descritivas                                     │
│  - Histogramas, boxplots, correlação                            │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 4: MODELAGEM E TREINAMENTO                               │
│  - Divisão: 80% treino / 20% teste                              │
│  - Algoritmos: Linear, Decision Tree, Random Forest, XGBoost    │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 5: VALIDAÇÃO E AVALIAÇÃO                                 │
│  - Métricas: R², MSE, MAE, RMSE                                 │
│  - Seleção do melhor modelo                                     │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 6: GERAÇÃO DE RELATÓRIO                                  │
│  - Relatório PDF com gráficos e métricas                        │
│  - Previsões prescritivas com ML                                │
└─────────────────────────────────────────────────────────────────┘
```

## Modelos Implementados

| Modelo           | Descrição           | Caso de Uso        |
|------------------|---------------------|--------------------|
| Regressão Linear | Baseline linear     | Referência         |
| Decision Tree    | Árvore de decisão   | Interpretabilidade |
| Random Forest    | Ensemble de árvores | Robustez           |
| XGBoost          | Gradient boosting   | Performance        |

## Métricas de Avaliação

| Métrica | Descrição                   |
|---------|-----------------------------|
| R²      | Coeficiente de determinação |
| MSE     | Erro quadrático médio       |
| MAE     | Erro absoluto médio         |
| RMSE    | Raiz do MSE                 |

## Dados

Este projeto utiliza **dados dummy gerados sinteticamente** para fins de demonstração. Nenhum dado real de produção está
incluído.

Os dados dummy simulam:
- 10 equipamentos (EQ-101 a EQ-110)
- Registros de produção com quantidades, refugo e retrabalho
- Dados de manutenção com medições de desgaste
- Período simulado de ~3 anos

## Licença

Demo Project - Open Source