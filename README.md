# DemoML — Pipeline de Machine Learning para Manutenção Prescritiva

Sistema de Machine Learning para **manutenção prescritiva** em equipamentos industriais de produção.

## Visão Geral

Pipeline completo de ciência de dados, da coleta de dados aos relatórios PDF/PPTX prescritivos. Prevê quantos dias faltam até a próxima manutenção de cada equipamento usando:

- Apontamentos de produção
- Medições de desgaste dos componentes A e B (substituídos a cada substituição)
- Estado atual do equipamento (produção acumulada, taxa de refugo, índice de desgaste)

**Equipamentos demo:** `EQ-101` … `EQ-127` (27 equipamentos sintéticos).

## Requisitos

- Python 3.8+ (testado em 3.12)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estrutura do Projeto

```
ml_pipeline_demo/
├── scripts/
│   ├── run_pipeline.py                     # Orquestrador (s00→s09)
│   ├── auto_pipeline.py                    # Detecção de mudanças (MD5) + watch
│   ├── history_manager.py                  # Histórico versionado
│   ├── s00_split_unified.py                # (opc) Separa arquivo único por equipamento
│   ├── s01_data_collection.py              # Coleta + integração
│   ├── s02_preprocessing.py                # Higienização + features (componente A/B)
│   ├── s03_eda.py                          # EDA básico
│   ├── s03b_advanced_eda.py                # (opc) EDA avançado
│   ├── s04_modeling.py                     # 4 algoritmos
│   ├── s05_evaluation.py                   # Métricas + seleção
│   ├── s06_generate_report.py              # PDF principal
│   ├── s07_cross_reference.py              # (opc) Cruzamentos histórico × produção
│   ├── s08_prescription.py                 # (opc) Fórmula prescritiva
│   └── s09_monthly_component_reports.py    # (opc) MD + PPTX por equipamento
├── config/
│   └── paths.py                            # Caminhos canônicos
├── data/
│   ├── raw/                                # EQ-*.csv (apontamentos)
│   ├── manutencao/                         # dados_manutencao.xlsx + historico_preventivas.xlsx
│   ├── arquivo_unico/                      # entradas para s00 (opcional)
│   ├── arquivo_unico_processado/           # arquivos já processados pelo s00
│   └── _reference_schema/                  # schemas de referência (não-input)
├── outputs/
│   ├── eda_plots/, models/, reports/, history/
│   ├── relatorios_mensais_componentes/         # EQ-*.md (saída do s09)
│   └── relatorios_mensais_componentes_ppt/     # EQ-*.pptx + consolidado + zip
├── generate_dummy_data.py                  # Gerador de dados sintéticos (27 equip.)
├── requirements.txt
├── CHANGELOG.md
└── GUIA_REPLICACAO.md                      # Blueprint para replicar o padrão
```

## Execução Rápida

### 1. Gerar dados sintéticos

```bash
python generate_dummy_data.py
```

Gera 27 arquivos `EQ-*.csv` em `data/raw/`, mais `dados_manutencao.xlsx` e `historico_preventivas.xlsx` em `data/manutencao/`.

### 2. Pipeline completo

```bash
cd scripts
python run_pipeline.py
```

### 3. Pipeline incremental (detecção de mudanças)

```bash
python scripts/auto_pipeline.py             # roda apenas se algum input mudou
python scripts/auto_pipeline.py --watch     # daemon (default 300 s)
python scripts/auto_pipeline.py --force     # ignora cache
```

### 4. Etapas isoladas

```bash
python scripts/run_pipeline.py --step 7     # cruzamentos
python scripts/run_pipeline.py --step 8     # prescrição
python scripts/run_pipeline.py --step 9     # relatórios mensais (MD + PPTX)
python scripts/run_pipeline.py --list       # lista todas as etapas
python scripts/run_pipeline.py --diagram    # diagrama
python scripts/run_pipeline.py --history    # histórico de execuções
```

## Fluxo do Pipeline

```
ETAPA 0  (opc)  Separação de arquivo único         → data/raw/EQ-*.csv
ETAPA 1         Coleta e integração                 → data_raw.csv
ETAPA 2         Pré-processamento + features        → data_preprocessed.csv, equipment_stats.{csv,json}
ETAPA 3         EDA básico                          → data_eda.csv, eda_plots/
ETAPA 3b (opc)  EDA avançado                        → eda_plots/
ETAPA 4         Modelagem (LR, DT, RF, XGBoost)     → models/*.joblib, train_test_split.npz
ETAPA 5         Avaliação                           → best_model.joblib, evaluation_report.txt
ETAPA 6  (opc)  Relatório PDF                       → Report_DemoML_RX.pdf
ETAPA 7  (opc)  Cruzamentos históricos              → historico_completo.csv, historico_recente.csv,
                                                       janelas_operacao.csv, ociosidade.csv
ETAPA 8  (opc)  Prescrição                          → prescricao_manutencao.csv
ETAPA 9  (opc)  Relatórios mensais por componente   → relatorios_mensais_componentes/EQ-*.md,
                                                       relatorios_mensais_componentes_ppt/EQ-*.pptx,
                                                       Apresentacao_Consolidada.pptx + .zip
```

## Fórmula Prescritiva (s08)

```text
T_base          = mediana(dias_em_operacao) por equipamento (fallback 450)
fator_desgaste  = clamp( 1 / (amplitude_atual / amplitude_mediana_hist) , 0.60 , 1.20 )
fator_massa     = clamp( 1 / (massa_atual / massa_mediana_janelas)      , 0.70 , 1.30 )
T_prescrito     = T_base × fator_desgaste × fator_massa
data_prescrita  = data_ultima_substituicao + T_prescrito + dias_ociosidade
```

Buckets de urgência sobre `dias_restantes = data_prescrita - hoje`:

| Bucket    | Faixa            |
|-----------|------------------|
| ATRASADO  | `< 0`            |
| URGENTE   | `0 ≤ x < 30`     |
| ATENÇÃO   | `30 ≤ x < 90`    |
| OK        | `≥ 90`           |

## Modelos Implementados

| Modelo           | Caso de Uso        |
|------------------|--------------------|
| Regressão Linear | Baseline           |
| Decision Tree    | Interpretabilidade |
| Random Forest    | Robustez           |
| XGBoost          | Performance        |

## Métricas

| Métrica | Descrição                   |
|---------|-----------------------------|
| R²      | Coeficiente de determinação |
| MSE     | Erro quadrático médio       |
| MAE     | Erro absoluto médio         |
| RMSE    | Raiz do MSE                 |

## Dados

Este projeto usa **dados sintéticos**. Nenhum dado real de produção está incluído nos inputs.
Os arquivos em `data/_reference_schema/` são apenas exemplos de schema (não-input do pipeline).

## Licença

Demo Project — Open Source
