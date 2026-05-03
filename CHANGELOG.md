# Changelog — DemoML Pipeline

Todas as alterações notáveis deste projeto são documentadas neste arquivo.

## [1.1.0] — 2026-05-01

### Adicionado

- **Etapa 7 (`s07_cross_reference.py`)**: cruzamentos histórico × produção. Gera
  `historico_completo.csv`, `historico_recente.csv`, `janelas_operacao.csv` e
  `ociosidade.csv`.
- **Etapa 8 (`s08_prescription.py`)**: aplicação da fórmula prescritiva
  `T_prescrito = T_base × fator_desgaste × fator_massa` com clamps `[0.60, 1.20]`
  e `[0.70, 1.30]`, fallback `T_base = 450 dias` e classificação em buckets
  ATRASADO / URGENTE / ATENÇÃO / OK.
- **Etapa 9 (`s09_monthly_component_reports.py`)**: gera relatórios mensais por
  equipamento em Markdown e PPTX (7 slides), apresentação consolidada e zip.
- **Diretório `data/_reference_schema/`** com 4 arquivos de referência
  (XLSX) que serviram apenas para alinhar o schema do gerador sintético.
- **`generate_dummy_data.py` reescrito**: 27 equipamentos `EQ-101…EQ-127`,
  schema de manutenção com header em 2 níveis, geração de
  `historico_preventivas.xlsx` e schema 100 % alinhado aos arquivos de referência.
- **`config/paths.py`**: novas constantes (`DATA_REFERENCE_SCHEMA_DIR`,
  `RELATORIOS_COMPONENTES_DIR`, `RELATORIOS_COMPONENTES_PPT_DIR`,
  `EDA_PLOTS_DIR`, `HISTORICO_*_FILE`, `PRESCRICAO_MANUTENCAO_FILE`).
- **Dependências**: `python-pptx>=0.6.21`, `pypdf>=4.0.0` em `requirements.txt`.

### Alterado

- Renomeação semântica `cilindro_*` → `componente_a_*` e `fuso_*` →
  `componente_b_*` em scripts, dummy generator e relatório PDF (terminologia neutra).
- `run_pipeline.py`: `PIPELINE_STEPS` expandido para 0–9 com etapas 6–9 marcadas
  como `optional=True` (núcleo crítico = 1–5).
- `auto_pipeline.py`: scan agora cobre `DadosProducao*.csv` além de `*.xlsx`.
- `GUIA_REPLICACAO_FASE02.md` renomeado para `GUIA_REPLICACAO.md`; referências a
  "SABO", "Fase02", "IJ-*", "RM.195", "extrusoras" substituídas por linguagem
  agnóstica de domínio.
- `README.md` reorganizado com novo fluxo, fórmula prescritiva e estrutura de
  diretórios atualizada.

### Removido

- Antigos artefatos `IJ-*.md`, `IJ-*.pptx`, `IJ-*.pdf` em `outputs/relatorios_mensais_componentes*/`
  passam a ser **gerados** pelo pipeline (Etapa 9) sob nomes `EQ-*`.
- Strings literais `"Cod_Produto_SAXXXXX"`, `"Equipamento_IJ_XXX"` e referências
  hardcoded a XLSX específicos.

## [1.0.0] — 2026-02-06

### Adicionado

- Pipeline de 6 etapas (s00–s06) para manutenção preditiva.
- Coleta e integração de dados de múltiplos equipamentos.
- Pré-processamento com engenharia de features.
- EDA básica e avançada.
- Modelagem com 4 algoritmos (Linear, Decision Tree, Random Forest, XGBoost).
- Avaliação e seleção automática do melhor modelo.
- Geração de relatório PDF com previsões.
- Detecção automática de alterações via hash MD5 (auto_pipeline).
- Modo watch para monitoramento contínuo.
- Script de geração de dados sintéticos (10 equipamentos demo iniciais).
