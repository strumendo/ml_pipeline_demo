# Changelog - DemoML Pipeline

Todas as alterações notáveis deste projeto serão documentadas neste arquivo.

## [1.0.0] - 2026-02-06

### Adicionado

#### Pipeline Completo de ML

- Pipeline de 6 etapas para manutenção prescritiva
- Coleta e integração de dados de múltiplos equipamentos
- Pré-processamento com engenharia de features
- Análise exploratória (EDA) básica e avançada
- Modelagem com 4 algoritmos (Linear, Decision Tree, Random Forest, XGBoost)
- Avaliação e seleção automática do melhor modelo
- Geração de relatório PDF com previsões

#### Previsão Prescritiva com Modelo ML

- Tabela de previsão histórica (baseada em intervalos)
- Tabela de previsão prescritiva (baseada no modelo ML)
- Comparação entre métodos (Histórico vs ML)
- Recomendações automáticas: "Antecipar", "Pode adiar", "Conforme"

#### Automação

- Detecção automática de alterações usando hash MD5
- Modo watch para monitoramento contínuo
- Reexecução automática do pipeline

#### Dados Dummy

- Script de geração de dados sintéticos
- 10 equipamentos demo (EQ-101 a EQ-110)
- Dados de produção e manutenção simulados

#### Gráficos

- Matriz de correlação completa
- Scatter plots coloridos por equipamento
- Resumo visual por equipamento
- Análise temporal e matriz de urgência