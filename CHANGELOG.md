# Changelog - DemoML Pipeline

Todas as alteracoes notaveis deste projeto serao documentadas neste arquivo.

## [1.0.0] - 2026-02-06

### Adicionado

#### Pipeline Completo de ML
- Pipeline de 6 etapas para manutencao prescritiva
- Coleta e integracao de dados de multiplos equipamentos
- Pre-processamento com engenharia de features
- Analise exploratoria (EDA) basica e avancada
- Modelagem com 4 algoritmos (Linear, Decision Tree, Random Forest, XGBoost)
- Avaliacao e selecao automatica do melhor modelo
- Geracao de relatorio PDF com previsoes

#### Previsao Prescritiva com Modelo ML
- Tabela de previsao historica (baseada em intervalos)
- Tabela de previsao prescritiva (baseada no modelo ML)
- Comparacao entre metodos (Historico vs ML)
- Recomendacoes automaticas: "Antecipar", "Pode adiar", "Conforme"

#### Automacao
- Deteccao automatica de alteracoes usando hash MD5
- Modo watch para monitoramento continuo
- Re-execucao automatica do pipeline

#### Dados Dummy
- Script de geracao de dados sinteticos
- 10 equipamentos demo (EQ-101 a EQ-110)
- Dados de producao e manutencao simulados

#### Graficos
- Matriz de correlacao completa
- Scatter plots coloridos por equipamento
- Resumo visual por equipamento
- Analise temporal e matriz de urgencia
