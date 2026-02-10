# DemoML - Pipeline de Machine Learning para Manutencao Prescritiva

Sistema de Machine Learning para **manutencao prescritiva** em equipamentos industriais de producao.

## Visao Geral

Este pipeline implementa um fluxo completo de ciencia de dados para manutencao prescritiva, desde a coleta de dados ate a geracao de relatorios PDF com **previsoes baseadas em ML**. O objetivo e prever quantos dias faltam ate a proxima manutencao de cada equipamento com base em:

- Dados historicos de producao
- **Medicoes de desgaste** (cilindro e fuso)
- **Estado atual do equipamento** (producao acumulada, taxa de refugo, indice de desgaste)

**Melhor Modelo:** XGBoost

**Equipamentos Monitorados:** 10 equipamentos demo (EQ-101 a EQ-110)

## Requisitos

### Sistema
- Python 3.8+
- Sistema operacional: Linux/Windows/macOS

### Dependencias Python

```bash
# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Instalar dependencias
pip install -r requirements.txt
```

## Estrutura do Projeto

```
ml_pipeline_demo/
├── scripts/                     # Scripts do pipeline
│   ├── run_pipeline.py          # Orquestrador principal
│   ├── auto_pipeline.py         # Automacao com deteccao de alteracoes
│   ├── s01_data_collection.py   # Etapa 1: Coleta e integracao
│   ├── s02_preprocessing.py     # Etapa 2: Pre-processamento
│   ├── s03_eda.py               # Etapa 3: Analise exploratoria
│   ├── s03b_advanced_eda.py     # Etapa 3b: EDA avancado
│   ├── s04_modeling.py          # Etapa 4: Modelagem
│   ├── s05_evaluation.py        # Etapa 5: Avaliacao
│   ├── s06_generate_report.py   # Etapa 6: Relatorio PDF
│   └── history_manager.py       # Gerenciador de historico
│
├── config/
│   └── paths.py                 # Configuracao centralizada de caminhos
│
├── data/
│   ├── manutencao/              # Dados de manutencao
│   │   └── dados_manutencao.csv # Historico de manutencao
│   └── raw/                     # Dados brutos de producao
│       └── EQ-*.csv             # Arquivos por equipamento
│
├── outputs/                     # Saidas do pipeline
│   ├── eda_plots/               # Graficos gerados
│   ├── models/                  # Modelos treinados
│   ├── history/                 # Historico de execucoes
│   └── Report_DemoML_RX.pdf     # Relatorio final
│
├── generate_dummy_data.py       # Script para gerar dados dummy
├── requirements.txt             # Dependencias Python
└── README.md
```

## Execucao Rapida

### 1. Gerar Dados Dummy (opcional - ja incluidos)

```bash
python generate_dummy_data.py
```

### 2. Executar Pipeline Completo

```bash
cd scripts
python run_pipeline.py
```

O pipeline executara todas as etapas automaticamente e gerara o relatorio PDF.

### 3. Execucao Automatica

```bash
# Verificar alteracoes e executar se necessario
python auto_pipeline.py

# Monitoramento continuo (verifica a cada 5 minutos)
python auto_pipeline.py --watch

# Forcar re-execucao
python auto_pipeline.py --force
```

### 4. Verificar Resultados

- Relatorio PDF: `outputs/Report_DemoML_RX.pdf`
- Metricas: `outputs/evaluation_report.txt`
- Graficos: `outputs/eda_plots/`

## Fluxo do Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  ETAPA 1: COLETA E INTEGRACAO                                   │
│  CSV Files (EQ-101, EQ-102, ...) → DataFrame Unico              │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 2: PRE-PROCESSAMENTO E LIMPEZA                           │
│  - Remocao de duplicatas, tratamento de nulos                   │
│  - Calculo de dias ate manutencao (variavel target)             │
│  - Features de medicao/desgaste                                 │
│  - One-Hot Encoding                                             │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 3: ANALISE EXPLORATORIA (EDA)                            │
│  - Estatisticas descritivas                                     │
│  - Histogramas, boxplots, correlacao                            │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 4: MODELAGEM E TREINAMENTO                               │
│  - Divisao: 80% treino / 20% teste                              │
│  - Algoritmos: Linear, Decision Tree, Random Forest, XGBoost    │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 5: VALIDACAO E AVALIACAO                                 │
│  - Metricas: R², MSE, MAE, RMSE                                 │
│  - Selecao do melhor modelo                                     │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 6: GERACAO DE RELATORIO                                  │
│  - Relatorio PDF com graficos e metricas                        │
│  - Previsoes prescritivas com ML                                │
└─────────────────────────────────────────────────────────────────┘
```

## Modelos Implementados

| Modelo | Descricao | Caso de Uso |
|--------|-----------|-------------|
| Regressao Linear | Baseline linear | Referencia |
| Decision Tree | Arvore de decisao | Interpretabilidade |
| Random Forest | Ensemble de arvores | Robustez |
| XGBoost | Gradient boosting | Performance |

## Metricas de Avaliacao

| Metrica | Descricao |
|---------|-----------|
| R² | Coeficiente de determinacao |
| MSE | Erro quadratico medio |
| MAE | Erro absoluto medio |
| RMSE | Raiz do MSE |

## Dados

Este projeto utiliza **dados dummy gerados sinteticamente** para fins de demonstracao. Nenhum dado real de producao esta incluido.

Os dados dummy simulam:
- 10 equipamentos (EQ-101 a EQ-110)
- Registros de producao com quantidades, refugo e retrabalho
- Dados de manutencao com medicoes de desgaste
- Periodo simulado de ~3 anos

## Licenca

Demo Project - Open Source

## Contato

Para duvidas ou sugestoes, consulte a development team.
