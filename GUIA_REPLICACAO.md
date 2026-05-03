# Guia de Replicação — Pipeline ML Prescritivo

> Documento-blueprint para **replicar o padrão arquitetural deste pipeline em outros projetos**.
> Foco em decisões, contratos e armadilhas — não em listar o que o código já faz.
> Leitura direcionada para um agente LLM que vá aplicar este desenho a um novo domínio.

---

## 0. O que é este pipeline (em uma frase)

Pipeline de **manutenção prescritiva** sobre equipamentos industriais (regressão: `dias_até_próxima_manutenção`), executado em **8 estágios sequenciais idempotentes** que se comunicam **por arquivos em `outputs/`** (não por objetos em memória), orquestrados por um `run_pipeline.py` que importa cada estágio como módulo, com **histórico versionado** por execução, **detecção automática de mudanças** nos dados de entrada e **relatório PDF final** com versão auto-incrementada.

A receita abaixo é **agnóstica ao domínio**: troque "manutenção", "equipamento" e "dias até próxima manutenção" por qualquer outro contexto e a estrutura se mantém.

---

## 1. Princípios arquiteturais (os mandamentos)

Os 8 princípios abaixo são o que torna esse pipeline reusável. Replicá-los é mais importante do que replicar o código.

### 1.1 Estágios como módulos com `main(**kwargs) -> dict`
Cada estágio é um arquivo `sXX_<nome>.py` exposto em **uma única função** `main(**pipeline_context) -> dict`. O orquestrador faz `__import__(script_name).main(**ctx)`. **Sempre aceitar `**kwargs`** — mesmo que o estágio ignore tudo —, porque flags novas (`--inicio`, `--fim`, `--version`, `--suffix`) fluem pelo `pipeline_context` e seriam quebradas a cada estágio adicionado.

### 1.2 Comunicação por arquivos, não por memória
O `return` de cada `main()` serve **só para o histórico**, nunca para alimentar o próximo estágio. O estágio seguinte **lê do disco**. Consequência: cada estágio é executável **isoladamente** (`run_pipeline.py --step 4`) sem precisar refazer os anteriores se os artefatos já existem. Trade-off aceito: I/O extra em troca de modularidade total.

### 1.3 `os.chdir(OUTPUTS_DIR)` no orquestrador
O `run_pipeline.py` muda o cwd para `outputs/` **antes** de chamar qualquer estágio. Isso permite que cada estágio escreva caminhos relativos curtos (`"data_raw.csv"`, `"models/"`) sem se preocupar com onde foi invocado. **Custo**: scripts **precisam** ser rodados a partir de `ML Pipeline/scripts/`, e qualquer caminho de input fora de `outputs/` precisa ser absoluto (ou via `paths.py`).

### 1.4 `paths.py` central com `ensure_directories()` no import
Um único módulo `config/paths.py` declara **todos** os diretórios e arquivos canônicos como constantes (`DATA_RAW_DIR`, `MODELS_DIR`, `BEST_MODEL_FILE`, …) e roda `ensure_directories()` no momento do `import` para criar a árvore. Isso elimina `os.makedirs(..., exist_ok=True)` espalhado pelo código.

### 1.5 Fallback de import "robusto"
Cada estágio tenta `from paths import ...` mas tem um **bloco de fallback** com caminhos relativos calculados a partir de `__file__`. Padrão:

```python
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
try:
    from paths import DATA_RAW_DIR, OUTPUTS_DIR, ...
except ImportError:
    DATA_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
    OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
    # ... funções helpers replicadas localmente
```

O fallback parece redundante, mas garante que o estágio rode `python sXX_*.py` direto, sem precisar do orquestrador na frente.

### 1.6 Declaração única do pipeline (single source of truth)
Um `dict` `PIPELINE_STEPS` em `run_pipeline.py` declara **ordem, título, descrição, inputs, outputs e flag `optional`** de cada etapa. `--list`, `--diagram`, dispatch, histórico e ranking de falhas leem desse dicionário. Nunca espalhe a ordem do pipeline em vários lugares.

### 1.7 Histórico versionado por `run_id` único
Cada execução recebe um `run_id = YYYYMMDD_HHMMSS` gerado **uma única vez** no orquestrador e propagado para `HistoryManager` **e** para o arquivo de log (`history/logs/run_<id>.log`). Isso casa logs, JSON estruturado e relatório textual da mesma execução. **Não gerar `run_id` dentro de cada estágio** — eles teriam timestamps diferentes.

### 1.8 Idempotência via hash MD5
O `auto_pipeline.py` calcula `MD5(arquivo)` para todo input rastreado e persiste em `outputs/.data_state.json`. Pipeline só roda se algum hash mudou. Isso permite cron a cada 5 min sem custo. **Nunca editar o `.data_state.json` à mão** — usar `--reset` ou `--force`.

---

## 2. Receita estágio-a-estágio (template)

Use esta tabela como ponto de partida ao replicar em outro domínio. A coluna **"papel genérico"** é o que você precisa entender.

| # | Stage DemoML | Papel genérico | Entrada típica | Saída típica | Optional? |
|---|------------|----------------|----------------|--------------|-----------|
| 0 | `s00_split_unified` | "Pré-ingestão": separa um arquivo agregado em arquivos por entidade | `data/arquivo_unico/*.xlsx` | `data/raw/<entidade>.{csv,xlsx}` + move original para `arquivo_unico_processado/` | ✅ |
| 1 | `s01_data_collection` | Coleta + integração + ISO normalize de datas | Vários arquivos heterogêneos em `data/raw/` | `outputs/data_raw.csv` | ❌ |
| 2 | `s02_preprocessing` | Higienização, **cálculo do target**, engenharia de features, OHE, **agregação ao grão correto** | `data_raw.csv` + `data/manutencao/*.xlsx` | `data_preprocessed.csv` + `equipment_stats.{csv,json}` | ❌ |
| 3 | `s03_eda` | EDA básico: estatísticas + gráficos canônicos | `data_preprocessed.csv` | `data_eda.csv` (igual + filtros), `eda_report.txt`, `eda_plots/` | ❌ |
| 3b | `s03b_advanced_eda` | EDA avançado, gráficos extras (correlação, urgência, scatter por entidade) | `data_eda.csv` | mais `eda_plots/*.png` | ✅ |
| 4 | `s04_modeling` | Split treino/teste **estratificado e fixo** + treino paralelo de N algoritmos | `data_eda.csv` | `models/*.joblib`, `train_test_split.npz` | ❌ |
| 5 | `s05_evaluation` | Avaliação no holdout + ranking + seleção do melhor | `models/`, `train_test_split.npz` | `best_model.joblib` (com metadados), `evaluation_report.txt` | ❌ |
| 6 | `s06_generate_report` | Relatório final (PDF/HTML/MD) com versão auto-incrementada | Tudo acima + `eda_plots/` | `Relatorio_<projeto>_R<N>.pdf` | ❌ (mas falha não derruba pipeline) |
| 7 | `s07_<x>` | Cruzamentos auxiliares (ex.: histórico vs produção) | Inputs cross-domain | CSVs específicos | ✅ |
| 8 | `s08_<prescricao>` | Prescrição derivada (ex.: "quando agir") combinando vários sinais | Saídas do s07 | `prescricao_<dominio>.csv` | ✅ |

**Regra:** se um estágio falha e está marcado como `optional=True` no `PIPELINE_STEPS`, o pipeline continua. Etapas críticas (1–5) **não** são opcionais. Etapa 6 é exceção: **falha no relatório nunca deve derrubar o resto** — métrica, modelo e CSVs ainda valem.

---

## 3. Contrato detalhado de cada estágio

Aqui descrevo o **mínimo viável** para replicar. Adapte os nomes ao seu domínio.

### 3.1 Estágio 1 — Coleta
- **Aceitar XLSX e CSV transparentemente.** Função `convert_xlsx_to_csv(filepath, force_refresh=True)` que lê o XLSX, converte coluna de data com `pd.to_datetime(..., dayfirst=True, errors='coerce')` e grava CSV com `date_format='%Y-%m-%d'`. **`force_refresh=True` por padrão** porque CSVs em cache ficam stale silenciosamente (já fomos mordidos por isso).
- **`find_data_directory()`** com fallback: primeiro tenta `DATA_RAW_DIR` do `paths.py`, depois lista de candidatos relativos. Retorna `None` se não achar nada.
- **`normalize_extended_format(df, equip_name)`** para arquivos com schema diferente (ex.: `EQ-138.2.xlsx`). Cuidado com `Path("EQ-138.2").with_suffix(".csv")` — o `with_suffix` interpreta `.2` como extensão e gera nome errado. Use `parent / f"{base_name}.csv"`.
- **Concatenar** todos os DataFrames com `pd.concat(..., ignore_index=True)`. Adicionar coluna `Fonte_Dados` apontando o arquivo de origem (útil para debug e dedup).

### 3.2 Estágio 2 — Pré-processamento (o estágio mais denso)
Ordem fixa de operações (não inverter):

1. `remove_duplicates(df)` — com base em chaves de negócio (ex.: data + equipamento + ordem), **não** em todas as colunas.
2. `handle_null_values(df)` — drop linhas sem chave; imputar numéricos com `0`/mediana conforme semântica.
3. `convert_dates(df)` — `pd.to_datetime(..., dayfirst=True, errors='coerce')`.
4. **`calculate_maintenance_days(df)` = cálculo do target.** Lógica de duas pernas:
   - registros **antes** da última manutenção: `target = data_manutencao_próxima - data_atual`
   - registros **depois** da última conhecida: `target = (ultima_data + intervalo_típico) - data_atual`
   Sem essa segunda perna, você descarta metade dos dados recentes.
5. `add_date_features(df)` — derivar `dia_da_semana`, `mes`, `dias_desde_inicio_serie`, `dias_desde_ultima_manutencao`. **Não esquecer dessa**: ausência de feature temporal é uma das três causas conhecidas de R² baixo neste pipeline (ver §7).
6. `generate_cumulative_variables(df)` — `Qtd_Produzida_Acumulado`, etc. Sempre `groupby(entidade).cumsum()` ordenado por data.
7. `add_measurement_features(df)` — features de "estado físico" do equipamento (medições, desgaste). **Atenção**: se essas features são **constantes por equipamento**, elas viram colineares com o OHE de equipamento (ver §7).
8. `add_maintenance_history_features(df)` e `add_preventive_history_features(df)` — features derivadas de planos de manutenção (planos preventivos externos).
9. `apply_one_hot_encoding(df)` — só categóricas com `nunique() <= 50`. Acima disso, pular para evitar explosão de dimensionalidade.
10. `clean_column_names(df)` — substituir espaços, pontos, parênteses, barras por `_`.
11. **`aggregate_by_day_equipment(df)` (CRÍTICO)** — múltiplas ordens do mesmo dia/equipamento têm **o mesmo target** (ruído irredutível). Agregar reduz overfitting. Regras por coluna:
    - `sum`: quantidades (`Qtd_Produzida`, `Qtd_Refugada`)
    - `mean`: taxas (`Fator_Un`, `Consumo_de_massa`)
    - `max`: acumulados (monotônicos no dia → last == max)
    - `max`: OHE de produto/massa/unidade (presença booleana)
    - `first`: constantes por equipamento (medições, OHE de equipamento) e por dia (target, features de data)
    - `drop`: metadados de linha (`Cod_Ordem`, `Fonte_Dados`)
12. **`calculate_equipment_statistics(df_stats)`** — recarrega o dataset **sem encoding** para gerar `equipment_stats.csv` legível por humano (com nomes de equipamento, não OHE).

> **Cache global em `_MAINTENANCE_CACHE`.** Manutenção é lida várias vezes (estatísticas, features, etc.). Ler XLSX é caro. Use `global _MAINTENANCE_CACHE` + invalidação **explícita** no início do `main()` (`_MAINTENANCE_CACHE = None`). Não deixar o cache "pegar carona" entre execuções via `--step`.

### 3.3 Estágio 3 — EDA
- `matplotlib.use('Agg')` **obrigatório** (backend não-interativo, senão trava em servidor headless).
- Gerar gráficos canônicos: `correlation_matrix_full.png`, `heatmap_correlacao.png`, `histogramas.png`, `boxplots.png`, `dispersao_target.png`, `scatter_plots_features.png` (colorido por entidade).
- `data_eda.csv` é geralmente **idêntico** ao `data_preprocessed.csv` — fica como artefato separado para isolar a fronteira "EDA terminado, pronto pra modelagem".

### 3.4 Estágio 4 — Modelagem
- **`prepare_features_target(df)` é onde o data leakage é eliminado.** Lista explícita `leaky_features = [...]` (ex.: `intervalo_manutencao`, qualquer coisa derivada do target). Lista `equipment_constant_features` que duplicam o OHE de equipamento (ver §7).
- `train_test_split(test_size=0.2, random_state=42)` — fixo. Sem CV elaborado nessa fase: a comparação cruzada entre 4 algoritmos no mesmo holdout já dá sinal suficiente.
- **4 algoritmos baseline** (sempre os mesmos, na mesma ordem):
  1. `LinearRegression` (sem hiperparâmetros — baseline puro)
  2. `DecisionTreeRegressor(max_depth=6, min_samples_split=10, min_samples_leaf=10)`
  3. `RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=10, min_samples_leaf=10, n_jobs=-1)`
  4. `XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1)` com fallback para `GradientBoostingRegressor` se XGBoost faltar.
  Os hiperparâmetros conservadores (`max_depth` baixo, `min_samples_leaf` alto) são **uma decisão deliberada** para evitar overfitting em dataset pequeno (~3k linhas pós-agregação). Não é "default" — é "default depois que apanhamos".
- `joblib.dump` cada modelo em `models/model_<nome>.joblib` + `np.savez(train_test_split.npz, X_train, X_test, y_train, y_test, feature_names)`.

### 3.5 Estágio 5 — Avaliação
- Métricas fixas: **R², MSE, MAE, RMSE.**
- Critério de seleção: **maior R²** (single objective). Se quiser trocar, expor parâmetro `criterion`.
- Salvar `best_model.joblib` como **`{"model": modelo, "name": nome, "metrics": {...}}`** — o relatório precisa do nome e das métricas, não só do modelo.
- `evaluation_report.txt` em texto puro, ordenado por R² desc, com `★` no melhor. Esse arquivo é lido no estágio 6 e por humanos.
- Auto-classificação: R² ≥ 0.9 → "excelente"; 0.7–0.9 → "bom"; <0.7 → "moderado/baixo". Útil no relatório.

### 3.6 Estágio 6 — Relatório
- **Auto-incremento de versão** lendo o último `Relatorio_*_R<N>.pdf` no diretório, extraindo `<N>` por regex e salvando como `R<N+1>`. Aceitar `--version` para fixar e `--suffix` para variantes (`_rascunho`, `_v1`).
- **Falha no relatório não derruba o pipeline** — modelo já está salvo, métricas no `evaluation_report.txt`, dados em CSVs.
- **Evitar embutir tabelas grandes no PDF.** O Capítulo 11 (R23+) deste projeto **substituiu 4 sub-tabelas por 2 links clicáveis** para CSV/PPTX. Resultado: PDF menor, dados consultáveis. Padrão a aplicar em qualquer relatório que cresça demais.
- Funções utilitárias que valem replicar:
  - `_load_sap_scheduled_dates()` — lê plano externo de manutenção e devolve `{entidade: [datas]}`.
  - `_load_ml_predictions()` — lê CSV do estágio 8 e devolve `{entidade: data_prevista}`.
  - `_load_last_production_dates()` — lê `data_raw.csv` e devolve `{entidade: max_data}`.
  - `generate_previsao_manutencao_csv(...)` — consolida 3 fontes (histórica × plano × ML) em um único CSV de 9 colunas.
  - `merge_componentes_pptx()` — concatena N PPTX em um único via `python-pptx` (cópia profunda de slides + relacionamentos de mídia).

### 3.7 Estágios 7 e 8 — Cruzamentos e prescrição (o "diferencial prescritivo")
Esses dois estágios são o que transforma um pipeline preditivo em **prescritivo**. Padrão geral:

- **s07** lê fontes brutas/históricas e gera **fotografias temporais** (`historico_completo.csv`) + **fotografia mais recente** (`historico_recente.csv`) + **janelas** (períodos entre eventos) + **ociosidade** (lacunas de inatividade).
  - Decisão importante: manter **todas** as leituras históricas no `historico_completo.csv` (mesma entidade pode aparecer várias vezes com `arquivo_origem` diferente). A "recente" é só uma view filtrada.
  - Para Excel serial dates: epoch é `1899-12-30` (Windows) — isso lida corretamente com o bug do `29/02/1900`.

- **s08** combina sinais com clamps de segurança:
  ```
  T_prescrito = T_base * fator_desgaste * fator_massa
  data_prescrita = data_ultima_acao + T_prescrito + dias_ociosidade
  ```
  - `T_base` = mediana histórica do equipamento; fallback fixo (450 dias) se histórico insuficiente.
  - `fator_*` clampeado em ranges conservadores (`[0.60, 1.20]` para desgaste, `[0.70, 1.30]` para consumo) — evita outliers em históricos curtos.
  - **Ociosidade soma direto à data prescrita** (máquina parada não desgasta). Negativos viram 0.
  - Output classificado em buckets de urgência (`ATRASADO`, `URGENTE` <30d, `ATENÇÃO` <90d, `OK`).

---

## 4. Flags do orquestrador (interface obrigatória)

Replique esse conjunto de flags **sem cortar nenhuma**. Cada uma resolve um cenário real de produção:

| Flag | Cenário |
|------|---------|
| `--step N` | Iterar em um estágio sem rodar tudo |
| `--list` | "Quais etapas existem?" |
| `--diagram` | Onboarding de um novo dev |
| `--history` | "O que rodou aqui?" |
| `--compare N` | Comparar últimas N execuções (tabela markdown) |
| `--no-history` | Testes "sujos" sem poluir histórico |
| `--no-log` | Pular o tee para `history/logs/run_<id>.log` |
| `--inicio` / `--fim` | Recortar período (flui via `pipeline_context`) |
| `--suffix` | Variante de relatório (`_rascunho`) |
| `--version` | Sobrescrever auto-incremento |

E para o `auto_pipeline.py`:

| Flag | Cenário |
|------|---------|
| `--status` | Diagnóstico ("o que mudou?") |
| `--force` | Forçar reprocessamento |
| `--watch` + `--interval` | Daemon |
| `--reset` | Limpar `.data_state.json` (corrupção) |

---

## 5. Histórico de execuções (não-negociável)

Para cada `run_id`:
- `outputs/history/runs/run_<id>.json` — estruturado, contém `steps.<nome>.models.<modelo>.{r2, mse, mae}` + `best_model` + `errors[]`.
- `outputs/history/reports/report_<id>.txt` — resumo textual humano.
- `outputs/history/logs/run_<id>.log` — `stdout`+`stderr` capturados via `_TeeStream` (escreve em `sys.__stdout__` E em arquivo, com `flush()` em ambos).

**Por que três formatos?**
- JSON: alimenta `--compare N` (`pd.DataFrame` + `to_markdown`).
- TXT: relatório humano após cada run.
- LOG: diagnóstico quando algo deu errado e a sessão de terminal já fechou.

---

## 6. Detecção de mudanças (auto_pipeline)

```python
hash_md5 = hashlib.md5()
with open(filepath, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
        hash_md5.update(chunk)
```

- Padrões de scan: `EQ-*.xlsx`, `EQ-*.csv`, `DadosProducao*.xlsx`, `Dados Manut*.xlsx`. **Adapte o pattern ao seu domínio**, mas mantenha o conceito: lista explícita, não `*.*`.
- Estado em `outputs/.data_state.json` (escondido, hidrofóbico ao git via `.gitignore`).
- Comparação: hash + tamanho + mtime. Qualquer divergência → reroda.
- `--watch` é loop infinito com `time.sleep(interval)`. Default 300s. Aceita `Ctrl+C` limpamente.

---

## 7. Armadilhas conhecidas (lições caras)

Estas são as três causas confirmadas de **R² baixo / overfitting** neste pipeline. Replique a mitigação no novo projeto.

### 7.1 Data leakage por feature derivada do target
A feature `intervalo_manutencao` (intervalo médio histórico) tinha correlação ~1.0 com o target → **R² = 1.0 falso**. Memorização, não aprendizado. **Mitigação**: lista explícita `leaky_features` em `s04_modeling.py:prepare_features_target()` que **remove pelo nome** antes do treino, com print listando o que foi cortado.

### 7.2 Features constantes por entidade vs OHE da entidade
Medições de componentes A/B são constantes por equipamento (vêm do XLSX de manutenção). Se você já tem `Equipamento_IJ_044`, `Equipamento_IJ_046`, …, **a medição não adiciona sinal** — ela é colinear com o OHE. Mantê-la causa só ruído de coeficiente. **Mitigação**: lista `equipment_constant_features` removida em `prepare_features_target()`.

### 7.3 Granularidade do target ≠ granularidade da linha
Múltiplas ordens de produção no mesmo dia/equipamento têm o mesmo `target` (dias até manutenção). Sem agregar, o modelo vê 30 cópias do mesmo target com `Qtd_Produzida` variando — pura noise. **Mitigação**: `aggregate_by_day_equipment()` em §3.2 passo 11.

### 7.4 Outras armadilhas operacionais
- **`Path.with_suffix(".csv")` em arquivos com ponto no nome** (`EQ-138.2.xlsx`) silenciosamente trunca o nome. Sempre usar `parent / f"{stem}.csv"` quando o stem pode conter `.`.
- **Não rodar de outro diretório.** Como o orquestrador faz `os.chdir(OUTPUTS_DIR)` e importa estágios por nome via `sys.path`, executar de `ML Pipeline/` em vez de `ML Pipeline/scripts/` quebra os imports.
- **Editar `.data_state.json` à mão** corrompe a detecção de mudanças. Use `--reset` ou `--force`.
- **Venv de outra arquitetura** ("cannot execute binary file: Exec format error"). Sempre recriar o venv localmente, nunca copiar entre máquinas.
- **Cache global de manutenção** (`_MAINTENANCE_CACHE`) precisa ser invalidado no início do `main()` da Etapa 2 — senão `--step 2` em sequência usa dados velhos.
- **dayfirst=True** em todos os `pd.to_datetime` — datas brasileiras (dd/mm/yyyy). `errors='coerce'` para não explodir em strings malformadas.

---

## 8. Convenções de código (manter ao replicar)

- **Idioma**: pt-BR em strings, docstrings, comentários, mensagens de CLI. Código em inglês (nomes de função, variáveis).
- **Cabeçalho fixo de cada estágio**: docstring com referência ao "fluxos.drawio" (ou seu equivalente), `sys.path.insert` para `config/`, import com fallback.
- **Saídas amigáveis**: `print` com `[N/M]` para progresso, `✓` para sucesso, `⚠` para warning, `✗` para erro. Caracteres unicode propositais — facilita scan visual no log.
- **Cwd durante run = `outputs/`**. Estágios escrevem caminhos relativos curtos.
- **`status` no return**. Cada `main()` retorna `{"status": "success" | "error", ...}`. O orquestrador checa.
- **Sem testes formais.** Não há `pytest`. A "validação" é o pipeline rodar end-to-end sem erro e o `evaluation_report.txt` ter R² razoável. Se você replicar isso em um projeto que precisa de testes, adicione — mas saiba que essa é a convenção atual.

---

## 9. Stack mínima (requirements)

```
pandas>=2.0          # DataFrame
numpy>=1.24          # Arrays
openpyxl>=3.1        # Leitura de XLSX
scikit-learn>=1.3    # LR, DT, RF, métricas, split
xgboost>=2.0         # gradient boosting (com fallback para sklearn)
matplotlib>=3.7      # plots
seaborn>=0.12        # heatmap, scatter estatísticos
reportlab>=4.0       # geração de PDF
pypdf>=4.0           # append/concat de PDFs
joblib>=1.3          # serialização de modelos
tabulate>=0.9        # df.to_markdown() para --compare
python-pptx          # se for replicar consolidação de PPTX
```

---

## 10. Checklist rápido para clonar este pattern em novo domínio

Use esta lista como guia de implementação:

1. [ ] Definir `BASE_DIR/data/{raw, manutencao, arquivo_unico, arquivo_unico_processado}` e `BASE_DIR/outputs/{models, plots, reports, history}`.
2. [ ] Escrever `config/paths.py` com constantes + `ensure_directories()` no import + `get_*_files()` helpers.
3. [ ] Definir o **target** e a **granularidade** (linha = quê?). Documentar em comentário no estágio 2.
4. [ ] Escrever `s01` com `convert_xlsx_to_csv` e `find_data_directory` com fallback.
5. [ ] Escrever `s02` com **as 12 sub-etapas em ordem** (§3.2). Especial atenção a target, agregação e cache.
6. [ ] Escrever `s03` com `matplotlib.use('Agg')` e os ~10 gráficos canônicos.
7. [ ] Escrever `s04` com 4 algoritmos e **listas explícitas de leaky_features e equipment_constant_features**.
8. [ ] Escrever `s05` com R²/MSE/MAE/RMSE + ranking + `best_model.joblib` empacotado com metadados.
9. [ ] Escrever `s06` com auto-incremento de versão e proteção contra falha total do pipeline.
10. [ ] (Opcional) `s07`/`s08` se o domínio tem componente prescritivo (cruzamentos + fórmula com clamps).
11. [ ] Escrever `run_pipeline.py` com `PIPELINE_STEPS` dict + dispatch por `__import__` + `_TeeStream` log + flags da §4.
12. [ ] Escrever `auto_pipeline.py` com hash MD5 + `.data_state.json`.
13. [ ] Escrever `history_manager.py` com `run_id` propagável e logs em 3 formatos.
14. [ ] **Rodar `--list`, `--diagram`, e o pipeline completo** antes de declarar feito.

---

## 11. O que NÃO copiar literalmente

- Os nomes `EQ-*`, "equipamento", "manutenção", "componentes A/B" — são do domínio do projeto. Adaptar.
- Os **valores hardcoded** de clamps (`0.60–1.20`, `0.70–1.30`) e fallback (`450 dias`) — calibrados para esse domínio. Recalibrar.
- O **sumário do PDF** (Capítulos 11–17, ordem específica) — é layout do domínio original. Reestruturar.
- A lista de equipamentos hardcoded em qualquer fallback — lê do XLSX de manutenção dinamicamente.
- Os patterns de scan (`EQ-*.xlsx`, `Dados Manut*.xlsx`) — substituir pelos do novo domínio.

---

## 12. Referências dentro deste repo

- Pipeline canônico: `/ML Pipeline/scripts/run_pipeline.py`
- Paths centrais: `/ML Pipeline/config/paths.py`
- Estágios: `/ML Pipeline/scripts/s00_*.py` … `s08_*.py`
- Histórico: `/ML Pipeline/outputs/history/{runs,reports,logs}/`
- Notas de R² baixo: `/ML Pipeline/MELHORIAS_MODELO_R2.md` (causas + fixes)
- Mudanças recentes: `/ML Pipeline/CHANGELOG.md`
- Padrões de relatório: `/ML Pipeline/scripts/s06_generate_report.py`
