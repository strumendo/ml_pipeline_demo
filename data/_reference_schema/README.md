# Schemas de referência

Os arquivos neste diretório **não são inputs do pipeline**. Servem apenas como referência de
**schema** (colunas, tipos, formatos) para o gerador de dados sintéticos
(`generate_dummy_data.py`) e para o desenvolvimento dos estágios.

O pipeline opera 100% sobre os dados sintéticos gerados em `data/raw/` e `data/manutencao/`
pelo gerador.

## Arquivos

| Arquivo | Schema referenciado |
|---------|---------------------|
| `manutencao_referencia.xlsx` | Histórico de manutenções por equipamento + medições dos componentes A (5 pontos + máx/mín) e B (4 pontos + máx/mín). Header em 2 níveis. |
| `historico_preventivas_referencia.xlsx` | Histórico de manutenções preventivas: Equipamento, GrpLisTar., Texto item man., Nº solicitação, Dta.iníc.progr., Ordem. |
| `producao_dez2025-jan2026_referencia.xlsx` | Apontamentos de produção (sheet `ag-grid`): Data, Cód. Ordem, Cód. Recurso, Cód. Produto, Qtd. Produzida/Refugada/Retrabalhada, Fator Un., Cód. Un., Descrição da massa (Composto), Consumo de massa. |
| `producao_fev-mar2026_referencia.xlsx` | Mesmo schema do anterior, período diferente. |

## Reprodutibilidade

Para gerar dados sintéticos novos com schema fiel ao destes arquivos:

```bash
python generate_dummy_data.py
```
