"""
S02 - Pré-processamento e Limpeza
=================================
Etapa 2 do Pipeline conforme fluxos.drawio

O QUE FAZ:
- Higienização e Transformação:
  - Remover duplicadas
  - Tratar valores nulos
  - Conversão Datas → datetime
- Engenharia de Features:
  - Geração de Variáveis Acumulativas
  - Codificação One-Hot
  - Features de medição de desgaste

FLUXO (fluxos.drawio):
DataFrame Único → Higienização → Engenharia de Features → Base para EDA

ENTRADA:
- data_raw.csv (saída da Etapa 1)

SAÍDA:
- data_preprocessed.csv: Dados limpos e transformados para EDA
- equipment_stats.csv: Estatísticas por equipamento

NOTA: Os dados de manutenção são carregados automaticamente do arquivo
      "Dados Manut*.xlsx" na pasta data/manutencao/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Adicionar config ao path
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR / "config"))

try:
    from paths import DATA_DIR, DATA_MANUTENCAO_DIR, get_maintenance_file, get_maintenance_history_file
except ImportError:
    DATA_DIR = BASE_DIR / "data"
    DATA_MANUTENCAO_DIR = DATA_DIR / "manutencao"

    def get_maintenance_file():
        if DATA_MANUTENCAO_DIR.exists():
            files = list(DATA_MANUTENCAO_DIR.glob("Dados Manut*.xlsx"))
            if files:
                return max(files, key=lambda f: f.stat().st_mtime)
        return None

    def get_maintenance_history_file():
        if DATA_MANUTENCAO_DIR.exists():
            files = list(DATA_MANUTENCAO_DIR.glob("*.csv"))
            if files:
                return max(files, key=lambda f: f.stat().st_mtime)
        return None

# Cache global para dados de manutenção (evita recarregar arquivo múltiplas vezes)
_MAINTENANCE_CACHE = None
_EQUIPMENT_STATS_CACHE = None


def load_maintenance_data() -> tuple:
    """
    Carrega dados de manutenção automaticamente do arquivo XLSX.

    Procura arquivos no padrão "Dados Manut*.xlsx" na pasta data/.

    Returns:
        Tupla (EQUIPAMENTO_MANUTENCAO, EQUIPAMENTO_INTERVALO)
    """
    global _MAINTENANCE_CACHE

    if _MAINTENANCE_CACHE is not None:
        return _MAINTENANCE_CACHE[:2]  # Retorna apenas manutencao e intervalo

    result = load_full_maintenance_data()
    return (result[0], result[1])


def load_full_maintenance_data() -> tuple:
    """
    Carrega dados completos de manutenção incluindo medições.

    Procura arquivos no padrão "Dados Manut*.xlsx" na pasta data/manutencao/
    ou data/ (fallback).

    Returns:
        Tupla (EQUIPAMENTO_MANUTENCAO, EQUIPAMENTO_INTERVALO, EQUIPAMENTO_MEDICOES)

    EQUIPAMENTO_MEDICOES contém para cada equipamento:
        - data_ultima_manutencao, data_penultima_manutencao
        - dias_operacao, observacoes
        - cilindro_a, cilindro_b, cilindro_c, cilindro_d, cilindro_e
        - cilindro_max, cilindro_min, cilindro_variacao
        - fuso_a, fuso_b, fuso_c, fuso_d
        - fuso_max, fuso_min, fuso_variacao
        - desgaste_cilindro (diferença max-min normalizada)
        - desgaste_fuso (diferença max-min normalizada)
    """
    global _MAINTENANCE_CACHE

    if _MAINTENANCE_CACHE is not None:
        return _MAINTENANCE_CACHE

    equipamento_manutencao = {}
    equipamento_intervalo = {}
    equipamento_medicoes = {}

    # Usar a função do paths.py para encontrar o arquivo
    maint_file = get_maintenance_file()

    if not maint_file:
        # Fallback: procurar diretamente
        for search_dir in [DATA_MANUTENCAO_DIR, DATA_DIR]:
            if search_dir.exists():
                maint_files = list(search_dir.glob("Dados Manut*.xlsx"))
                if maint_files:
                    maint_file = max(maint_files, key=lambda f: f.stat().st_mtime)
                    break

    if not maint_file:
        print("  ⚠ Arquivo de manutenção não encontrado")
        print("    Procurado em: data/manutencao/ e data/")
        print("    Usando valores padrão.")
        _MAINTENANCE_CACHE = (_get_default_maintenance(), _get_default_intervals(), {})
        return _MAINTENANCE_CACHE

    print(f"  📋 Carregando dados de manutenção: {maint_file.name}")

    try:
        # Ler arquivo Excel
        df = pd.read_excel(maint_file, header=None)

        # Estrutura do arquivo:
        # Coluna 1: Equipamento (EQ-XXX)
        # Coluna 2: Data execução da última substituição
        # Coluna 3: Data da penúltima substituição
        # Coluna 4: Dias em operação
        # Coluna 5: Observações
        # Colunas 6-12: Medições Cilindro (A, B, C, D, E, Máximo, Mínimo)
        # Colunas 13-18: Medições Fuso (A, B, C, D, Máximo, Mínimo)

        for idx, row in df.iterrows():
            if idx < 2:  # Pular cabeçalhos
                continue

            equipamento = row[1]  # Coluna B
            data_ultima = row[2]  # Coluna C
            data_penultima = row[3]  # Coluna D - Data da penúltima substituição
            dias_operacao = row[4]  # Coluna E
            observacoes = row[5]  # Coluna F - Observações

            # Validar equipamento
            if pd.isna(equipamento) or not str(equipamento).startswith("EQ-"):
                continue

            equipamento = str(equipamento).strip()

            # Data da última manutenção
            data_ultima_str = None
            if pd.notna(data_ultima):
                try:
                    data_ultima_dt = pd.to_datetime(data_ultima)
                    data_ultima_str = data_ultima_dt.strftime("%Y-%m-%d")
                    equipamento_manutencao[equipamento] = data_ultima_str
                except Exception:
                    pass

            # Data da penúltima manutenção
            data_penultima_str = None
            if pd.notna(data_penultima):
                try:
                    data_penultima_dt = pd.to_datetime(data_penultima)
                    data_penultima_str = data_penultima_dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

            # Intervalo de operação
            if pd.notna(dias_operacao):
                try:
                    equipamento_intervalo[equipamento] = int(dias_operacao)
                except (ValueError, TypeError):
                    equipamento_intervalo[equipamento] = 365

            # Iniciar dicionário de medições com datas e observações
            medicoes = {
                "data_ultima_manutencao": data_ultima_str,
                "data_penultima_manutencao": data_penultima_str,
                "dias_operacao": int(dias_operacao) if pd.notna(dias_operacao) else None,
                "observacoes": str(observacoes) if pd.notna(observacoes) else None,
            }

            # Cilindro
            cil_a = _safe_float(row[6])
            cil_b = _safe_float(row[7])
            cil_c = _safe_float(row[8])
            cil_d = _safe_float(row[9])
            cil_e = _safe_float(row[10])
            cil_max = _safe_float(row[11])
            cil_min = _safe_float(row[12])

            medicoes["cilindro_a"] = cil_a
            medicoes["cilindro_b"] = cil_b
            medicoes["cilindro_c"] = cil_c
            medicoes["cilindro_d"] = cil_d
            medicoes["cilindro_e"] = cil_e
            medicoes["cilindro_max"] = cil_max
            medicoes["cilindro_min"] = cil_min

            # Calcular variação e desgaste do cilindro
            if cil_max is not None and cil_min is not None:
                medicoes["cilindro_variacao"] = cil_max - cil_min
                # Desgaste normalizado (quanto maior, mais desgastado)
                # Valor nominal do cilindro é ~20mm
                medicoes["desgaste_cilindro"] = (cil_max - 20.0) if cil_max else 0.0
            else:
                medicoes["cilindro_variacao"] = None
                medicoes["desgaste_cilindro"] = None

            # Fuso (colunas 13-18)
            fuso_a = _safe_float(row[13])
            fuso_b = _safe_float(row[14])
            fuso_c = _safe_float(row[15])
            fuso_d = _safe_float(row[16])
            fuso_max = _safe_float(row[17])
            fuso_min = _safe_float(row[18])

            medicoes["fuso_a"] = fuso_a
            medicoes["fuso_b"] = fuso_b
            medicoes["fuso_c"] = fuso_c
            medicoes["fuso_d"] = fuso_d
            medicoes["fuso_max"] = fuso_max
            medicoes["fuso_min"] = fuso_min

            # Calcular variação e desgaste do fuso
            if fuso_max is not None and fuso_min is not None:
                medicoes["fuso_variacao"] = fuso_max - fuso_min
                # Desgaste do fuso (quanto menor em relação a 20mm, mais desgastado)
                medicoes["desgaste_fuso"] = (20.0 - fuso_min) if fuso_min else 0.0
            else:
                medicoes["fuso_variacao"] = None
                medicoes["desgaste_fuso"] = None

            equipamento_medicoes[equipamento] = medicoes

        print(f"    ✓ Carregados {len(equipamento_manutencao)} equipamentos")
        equip_com_medicoes = sum(1 for m in equipamento_medicoes.values()
                                  if m.get("cilindro_a") is not None or m.get("fuso_a") is not None)
        print(f"    ✓ Equipamentos com medições: {equip_com_medicoes}")

    except Exception as e:
        print(f"  ⚠ Erro ao ler arquivo de manutenção: {e}")
        print("    Usando valores padrão.")
        _MAINTENANCE_CACHE = (_get_default_maintenance(), _get_default_intervals(), {})
        return _MAINTENANCE_CACHE

    # Usar valores padrão para equipamentos não encontrados
    default_maint = _get_default_maintenance()
    default_int = _get_default_intervals()

    for equip in default_maint:
        if equip not in equipamento_manutencao:
            equipamento_manutencao[equip] = default_maint[equip]
        if equip not in equipamento_intervalo:
            equipamento_intervalo[equip] = default_int.get(equip, 365)

    _MAINTENANCE_CACHE = (equipamento_manutencao, equipamento_intervalo, equipamento_medicoes)
    return _MAINTENANCE_CACHE


def _safe_float(value) -> float:
    """Converte valor para float de forma segura."""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _get_default_maintenance() -> dict:
    """Retorna valores padrão de manutenção (fallback)."""
    return {
        "EQ-101": "2024-03-15",
        "EQ-102": "2024-06-20",
        "EQ-103": "2024-09-10",
        "EQ-104": "2024-07-05",
        "EQ-105": "2024-06-28",
        "EQ-106": "2024-08-15",
        "EQ-107": "2024-10-01",
        "EQ-108": "2024-09-20",
        "EQ-109": "2024-07-01",
        "EQ-110": "2024-05-10",
        "EQ-111": "2024-04-25",
        "EQ-112": "2024-08-30",
        "EQ-113": "2025-01-15",
        "EQ-114": "2024-06-20",
        "EQ-115": "2024-08-05",
        "EQ-116": "2024-10-15",
        "EQ-117": "2024-11-20",
        "EQ-118": "2024-10-20",
        "EQ-119": "2025-02-10",
        "EQ-120": "2024-12-15",
        "EQ-121": "2024-12-15",
        "EQ-122": "2024-06-15",
        "EQ-123": "2025-01-05",
        "EQ-124": "2024-11-25",
        "EQ-125": "2024-07-10",
        "EQ-126": "2025-01-20",
        "EQ-127": "2024-08-10",
    }


def _get_default_intervals() -> dict:
    """Retorna valores padrão de intervalos (fallback)."""
    return {
        "EQ-101": 365,
        "EQ-102": 343,
        "EQ-103": 496,
        "EQ-104": 385,
        "EQ-105": 379,
        "EQ-106": 406,
        "EQ-107": 490,
        "EQ-108": 448,
        "EQ-109": 384,
        "EQ-110": 332,
        "EQ-111": 395,
        "EQ-112": 433,
        "EQ-113": 504,
        "EQ-114": 356,
        "EQ-115": 419,
        "EQ-116": 490,
        "EQ-117": 523,
        "EQ-118": 492,
        "EQ-119": 532,
        "EQ-120": 538,
        "EQ-121": 539,
        "EQ-122": 342,
        "EQ-123": 569,
        "EQ-124": 526,
        "EQ-125": 357,
        "EQ-126": 510,
        "EQ-127": 406,
    }


# Variáveis globais carregadas dinamicamente
# (mantidas para compatibilidade, mas recomenda-se usar load_maintenance_data())
EQUIPAMENTO_MANUTENCAO = _get_default_maintenance()
EQUIPAMENTO_INTERVALO = _get_default_intervals()


def add_measurement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features de medições de desgaste (cilindro e fuso) ao DataFrame.

    Carrega as medições do arquivo de manutenção e as incorpora como features
    para cada equipamento. Também calcula features derivadas como:
    - Taxa de desgaste estimada por peça produzida
    - Índice de urgência baseado em desgaste e produção acumulada

    Args:
        df: DataFrame de entrada

    Returns:
        DataFrame com features de medição adicionadas
    """
    # Carregar dados completos de manutenção incluindo medições
    _, equip_intervalo, equip_medicoes = load_full_maintenance_data()

    if not equip_medicoes:
        print("  ⚠ Sem dados de medições disponíveis")
        return df

    # Identificar coluna de equipamento
    equip_col = None
    for col in ["Equipamento", "Cod Recurso"]:
        if col in df.columns:
            equip_col = col
            break

    if equip_col is None:
        print("  ⚠ Coluna de equipamento não encontrada")
        return df

    print("  Adicionando features de medições de desgaste...")

    # Features de medição a adicionar
    measurement_features = [
        "cilindro_max", "cilindro_min", "cilindro_variacao", "desgaste_cilindro",
        "fuso_max", "fuso_min", "fuso_variacao", "desgaste_fuso"
    ]

    # Adicionar colunas de medição
    for feature in measurement_features:
        df[feature] = df[equip_col].apply(
            lambda x: equip_medicoes.get(x, {}).get(feature)
        )

    # Calcular features derivadas

    # 1. Intervalo médio de operação do equipamento
    df["intervalo_manutencao"] = df[equip_col].apply(
        lambda x: equip_intervalo.get(x, 365)
    )

    # 2. Taxa de desgaste estimada do cilindro por dia
    #    (desgaste / dias de operação)
    df["taxa_desgaste_cilindro"] = df.apply(
        lambda row: (row["desgaste_cilindro"] / row["intervalo_manutencao"])
        if pd.notna(row["desgaste_cilindro"]) and row["intervalo_manutencao"] > 0
        else 0.0,
        axis=1
    )

    # 3. Taxa de desgaste estimada do fuso por dia
    df["taxa_desgaste_fuso"] = df.apply(
        lambda row: (row["desgaste_fuso"] / row["intervalo_manutencao"])
        if pd.notna(row["desgaste_fuso"]) and row["intervalo_manutencao"] > 0
        else 0.0,
        axis=1
    )

    # 4. Índice de desgaste combinado (média ponderada cilindro + fuso)
    df["indice_desgaste"] = df.apply(
        lambda row: _calc_indice_desgaste(row),
        axis=1
    )

    # 5. Se temos quantidade produzida acumulada, calcular desgaste por peça
    qty_col = None
    for col in ["Qtd_Produzida_Acumulado", "Qtd. Produzida"]:
        if col in df.columns:
            qty_col = col
            break

    if qty_col:
        # Taxa de desgaste por 1000 peças produzidas
        df["desgaste_por_1000_pecas"] = df.apply(
            lambda row: _calc_desgaste_por_pecas(row, qty_col, equip_medicoes),
            axis=1
        )

    # Preencher valores nulos de medição com a média do grupo
    for feature in measurement_features + ["indice_desgaste", "taxa_desgaste_cilindro", "taxa_desgaste_fuso"]:
        if feature in df.columns:
            median_val = df[feature].median()
            if pd.notna(median_val):
                df[feature] = df[feature].fillna(median_val)
            else:
                df[feature] = df[feature].fillna(0.0)

    features_added = len(measurement_features) + 5  # medições + derivadas
    print(f"  ✓ Adicionadas {features_added} features de medição/desgaste")

    return df


def _calc_indice_desgaste(row) -> float:
    """
    Calcula índice de desgaste combinado (0-100).

    O índice considera:
    - Desgaste do cilindro (peso 60%)
    - Desgaste do fuso (peso 40%)

    Valores maiores indicam maior urgência de manutenção.
    """
    desgaste_cil = row.get("desgaste_cilindro")
    desgaste_fuso = row.get("desgaste_fuso")

    # Normalizar para escala 0-100
    # Desgaste cilindro: 0-0.6mm típico → 0-100
    # Desgaste fuso: 0-2mm típico → 0-100

    score_cil = 0.0
    score_fuso = 0.0

    if pd.notna(desgaste_cil):
        score_cil = min(100, (desgaste_cil / 0.6) * 100)

    if pd.notna(desgaste_fuso):
        score_fuso = min(100, (desgaste_fuso / 2.0) * 100)

    # Peso: 60% cilindro, 40% fuso
    return (score_cil * 0.6) + (score_fuso * 0.4)


def _calc_desgaste_por_pecas(row, qty_col: str, equip_medicoes: dict) -> float:
    """
    Calcula taxa de desgaste por 1000 peças produzidas.

    Esta métrica ajuda a prever manutenção baseada na produção,
    não apenas no tempo.
    """
    equip = row.get("Equipamento") or row.get("Cod Recurso")
    qty_acum = row.get(qty_col, 0)

    if not equip or qty_acum <= 0:
        return 0.0

    medicoes = equip_medicoes.get(equip, {})
    desgaste_total = 0.0

    desg_cil = medicoes.get("desgaste_cilindro")
    desg_fuso = medicoes.get("desgaste_fuso")

    if pd.notna(desg_cil):
        desgaste_total += desg_cil
    if pd.notna(desg_fuso):
        desgaste_total += desg_fuso

    # Taxa por 1000 peças
    return (desgaste_total / qty_acum) * 1000


def calculate_equipment_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas agregadas por equipamento.

    Gera um DataFrame com métricas detalhadas por equipamento para
    inclusão no relatório final.

    Args:
        df: DataFrame preprocessado

    Returns:
        DataFrame com estatísticas por equipamento
    """
    global _EQUIPMENT_STATS_CACHE

    # Identificar coluna de equipamento
    equip_col = None
    for col in ["Equipamento", "Cod Recurso", "Cód_Recurso"]:
        if col in df.columns:
            equip_col = col
            break

    if equip_col is None:
        print("  ⚠ Coluna de equipamento não encontrada para estatísticas")
        return pd.DataFrame()

    print("  Calculando estatísticas por equipamento...")

    # Carregar dados de manutenção
    _, equip_intervalo, equip_medicoes = load_full_maintenance_data()

    # Definir colunas para agregação
    agg_dict = {}

    # Quantidade produzida
    for col in ["Qtd_Produzida", "Qtd. Produzida"]:
        if col in df.columns:
            agg_dict[col] = ["sum", "mean", "max", "count"]
            break

    # Quantidade produzida acumulada
    for col in ["Qtd_Produzida_Acumulado", "Qtd. Produzida_Acumulado"]:
        if col in df.columns:
            agg_dict[col] = ["max"]
            break

    # Quantidade refugada
    for col in ["Qtd_Refugada", "Qtd. Refugada"]:
        if col in df.columns:
            agg_dict[col] = ["sum", "mean"]
            break

    # Quantidade retrabalhada
    for col in ["Qtd_Retrabalhada", "Qtd. Retrabalhada"]:
        if col in df.columns:
            agg_dict[col] = ["sum", "mean"]
            break

    # Consumo de massa
    for col in ["Consumo_de_massa_no_item_em_Kg_100pçs", "Consumo de massa no item em (Kg/100pçs)"]:
        if col in df.columns:
            agg_dict[col] = ["sum", "mean"]
            break

    # Manutenção (target)
    if "Manutencao" in df.columns:
        agg_dict["Manutencao"] = ["mean", "min", "max"]

    # Features de desgaste
    for col in ["indice_desgaste", "desgaste_cilindro", "desgaste_fuso"]:
        if col in df.columns:
            agg_dict[col] = ["mean"]

    if not agg_dict:
        print("  ⚠ Nenhuma coluna disponível para agregação")
        return pd.DataFrame()

    # Calcular agregações
    stats = df.groupby(equip_col).agg(agg_dict)

    # Achatar nomes das colunas
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()

    # Adicionar informações de manutenção
    stats["intervalo_manutencao_dias"] = stats[equip_col].apply(
        lambda x: equip_intervalo.get(x, None)
    )

    stats["data_ultima_manutencao"] = stats[equip_col].apply(
        lambda x: equip_medicoes.get(x, {}).get("data_ultima_manutencao")
    )

    stats["data_penultima_manutencao"] = stats[equip_col].apply(
        lambda x: equip_medicoes.get(x, {}).get("data_penultima_manutencao")
    )

    stats["observacoes_manutencao"] = stats[equip_col].apply(
        lambda x: equip_medicoes.get(x, {}).get("observacoes")
    )

    # Adicionar medições
    stats["cilindro_max"] = stats[equip_col].apply(
        lambda x: equip_medicoes.get(x, {}).get("cilindro_max")
    )

    stats["cilindro_min"] = stats[equip_col].apply(
        lambda x: equip_medicoes.get(x, {}).get("cilindro_min")
    )

    stats["fuso_max"] = stats[equip_col].apply(
        lambda x: equip_medicoes.get(x, {}).get("fuso_max")
    )

    stats["fuso_min"] = stats[equip_col].apply(
        lambda x: equip_medicoes.get(x, {}).get("fuso_min")
    )

    # Calcular taxa de refugo
    qty_sum_col = next((c for c in stats.columns if "Produzida_sum" in c), None)
    ref_sum_col = next((c for c in stats.columns if "Refugada_sum" in c), None)

    if qty_sum_col and ref_sum_col:
        stats["taxa_refugo_pct"] = (
            stats[ref_sum_col] / stats[qty_sum_col] * 100
        ).round(2)

    # Renomear colunas para melhor legibilidade
    rename_map = {
        equip_col: "equipamento",
    }

    # Renomear colunas específicas
    for old_col in stats.columns:
        if "Produzida_sum" in old_col:
            rename_map[old_col] = "total_produzido"
        elif "Produzida_mean" in old_col:
            rename_map[old_col] = "media_producao_diaria"
        elif "Produzida_max" in old_col and "Acumulado" not in old_col:
            rename_map[old_col] = "max_producao_diaria"
        elif "Produzida_count" in old_col:
            rename_map[old_col] = "total_registros"
        elif "Acumulado_max" in old_col:
            rename_map[old_col] = "producao_acumulada"
        elif "Refugada_sum" in old_col:
            rename_map[old_col] = "total_refugado"
        elif "Refugada_mean" in old_col:
            rename_map[old_col] = "media_refugo_diario"
        elif "Retrabalhada_sum" in old_col:
            rename_map[old_col] = "total_retrabalhado"
        elif "Retrabalhada_mean" in old_col:
            rename_map[old_col] = "media_retrabalho_diario"
        elif "Consumo" in old_col and "sum" in old_col:
            rename_map[old_col] = "consumo_massa_total_kg"
        elif "Consumo" in old_col and "mean" in old_col:
            rename_map[old_col] = "consumo_massa_medio_kg"
        elif "Manutencao_mean" in old_col:
            rename_map[old_col] = "media_dias_manutencao"
        elif "Manutencao_min" in old_col:
            rename_map[old_col] = "min_dias_manutencao"
        elif "Manutencao_max" in old_col:
            rename_map[old_col] = "max_dias_manutencao"
        elif "indice_desgaste_mean" in old_col:
            rename_map[old_col] = "indice_desgaste_medio"
        elif "desgaste_cilindro_mean" in old_col:
            rename_map[old_col] = "desgaste_cilindro_medio"
        elif "desgaste_fuso_mean" in old_col:
            rename_map[old_col] = "desgaste_fuso_medio"

    stats = stats.rename(columns=rename_map)

    # Ordenar por equipamento
    if "equipamento" in stats.columns:
        stats = stats.sort_values("equipamento")

    # Salvar em cache
    _EQUIPMENT_STATS_CACHE = stats

    print(f"  ✓ Estatísticas calculadas para {len(stats)} equipamentos")

    return stats


def get_equipment_statistics() -> pd.DataFrame:
    """
    Retorna estatísticas de equipamento do cache ou arquivo.

    Returns:
        DataFrame com estatísticas por equipamento
    """
    global _EQUIPMENT_STATS_CACHE

    if _EQUIPMENT_STATS_CACHE is not None:
        return _EQUIPMENT_STATS_CACHE

    # Tentar carregar do arquivo
    stats_file = Path("equipment_stats.csv")
    if stats_file.exists():
        return pd.read_csv(stats_file)

    return pd.DataFrame()


def export_equipment_statistics(stats: pd.DataFrame, output_path: str = "equipment_stats.csv"):
    """
    Exporta estatísticas de equipamento para CSV e JSON.

    Args:
        stats: DataFrame com estatísticas
        output_path: Caminho do arquivo CSV de saída
    """
    if stats.empty:
        return

    # Salvar CSV
    stats.to_csv(output_path, index=False)
    print(f"  ✓ Estatísticas salvas em: {output_path}")

    # Salvar JSON para uso no relatório
    json_path = output_path.replace(".csv", ".json")
    stats_dict = stats.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✓ Estatísticas salvas em: {json_path}")


def load_raw_data(filepath: str = "data_raw.csv") -> pd.DataFrame:
    """
    Carrega dados brutos da Etapa 1.

    Args:
        filepath: Caminho do arquivo CSV

    Returns:
        DataFrame com dados brutos
    """
    df = pd.read_csv(filepath)
    print(f"  Carregado: {len(df)} registros, {len(df.columns)} colunas")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove registros duplicados.

    Args:
        df: DataFrame de entrada

    Returns:
        DataFrame sem duplicatas
    """
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)

    if removed > 0:
        print(f"  ✓ Removidas {removed} duplicatas ({initial_count} → {len(df)})")
    else:
        print(f"  ✓ Nenhuma duplicata encontrada")

    return df


def handle_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores nulos no DataFrame.

    Estratégia:
    - Colunas numéricas: preenche com mediana
    - Colunas categóricas: preenche com moda ou 'Desconhecido'

    Args:
        df: DataFrame de entrada

    Returns:
        DataFrame com nulos tratados
    """
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls == 0:
        print(f"  ✓ Nenhum valor nulo encontrado")
        return df

    print(f"  Tratando {total_nulls} valores nulos...")

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                # Numérico: preencher com mediana
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"    {col}: preenchido com mediana ({median_val:.2f})")
            else:
                # Categórico: preencher com moda ou 'Desconhecido'
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                    print(f"    {col}: preenchido com moda ({mode_val[0]})")
                else:
                    df[col] = df[col].fillna("Desconhecido")
                    print(f"    {col}: preenchido com 'Desconhecido'")

    print(f"  ✓ Valores nulos tratados")
    return df


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas de data para datetime.

    Args:
        df: DataFrame de entrada

    Returns:
        DataFrame com datas convertidas
    """
    date_columns = [col for col in df.columns if "Data" in col or "data" in col]

    for col in date_columns:
        if col in df.columns and df[col].dtype == 'object':
            try:
                # Tentar formato brasileiro primeiro (dd/mm/yyyy)
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                print(f"  ✓ Convertido {col} para datetime")
            except Exception as e:
                print(f"  ⚠ Erro ao converter {col}: {e}")

    return df


def calculate_maintenance_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula dias até a próxima manutenção.

    Carrega dados de manutenção automaticamente do arquivo XLSX
    e calcula a variável target 'Manutencao' (dias restantes).

    Para registros após a última manutenção conhecida, calcula a próxima
    manutenção prevista usando o intervalo médio do equipamento.

    Args:
        df: DataFrame de entrada

    Returns:
        DataFrame com coluna de manutenção
    """
    # Carregar dados de manutenção dinamicamente
    equip_manutencao, equip_intervalo = load_maintenance_data()

    # Identificar coluna de data
    date_col = None
    for col in ["Data de Produção", "Data de Produção Acumulada"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        print("  ⚠ Coluna de data não encontrada. Gerando target sintético.")
        # Gerar target sintético baseado em outras features
        df["Manutencao"] = np.random.randint(1, 365, size=len(df))
        return df

    # Identificar coluna de equipamento
    equip_col = None
    for col in ["Equipamento", "Cod Recurso"]:
        if col in df.columns:
            equip_col = col
            break

    if equip_col is None:
        print("  ⚠ Coluna de equipamento não encontrada. Usando data fixa.")
        default_maint_date = pd.to_datetime("2024-06-01")
        df["Manutencao"] = (default_maint_date - pd.to_datetime(df[date_col])).dt.days
    else:
        # Calcular dias até manutenção por equipamento
        def calc_days(row):
            equip = row[equip_col]
            prod_date = pd.to_datetime(row[date_col])

            if equip in equip_manutencao:
                maint_date = pd.to_datetime(equip_manutencao[equip])
                intervalo = equip_intervalo.get(equip, 365)

                # Se a data de produção é posterior à última manutenção,
                # calcular a próxima manutenção prevista
                if prod_date > maint_date:
                    # Calcular próxima manutenção = última manutenção + intervalo
                    next_maint = maint_date + pd.Timedelta(days=intervalo)
                    return (next_maint - prod_date).days
                else:
                    return (maint_date - prod_date).days
            else:
                # Equipamento não mapeado - usar data default
                maint_date = pd.to_datetime("2024-06-01")
                return (maint_date - prod_date).days

        df["Manutencao"] = df.apply(calc_days, axis=1)

    # Remover registros com Manutencao negativa (após manutenção prevista)
    initial_count = len(df)
    df = df[df["Manutencao"] >= 0]

    if len(df) < initial_count:
        print(f"  ✓ Removidos {initial_count - len(df)} registros pós-manutenção")

    print(f"  ✓ Calculada variável 'Manutencao' (dias até manutenção)")

    return df


def generate_cumulative_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera variáveis acumulativas por equipamento.

    Calcula acumulados de:
    - Quantidade produzida
    - Quantidade refugada
    - Quantidade retrabalhada
    - Consumo de massa

    Args:
        df: DataFrame de entrada

    Returns:
        DataFrame com variáveis acumulativas
    """
    # Identificar colunas de quantidade
    qty_cols = [col for col in df.columns if any(x in col.lower() for x in ["qtd", "quantidade", "consumo"])]

    if not qty_cols:
        print("  ⚠ Nenhuma coluna de quantidade encontrada para acumular")
        return df

    # Identificar coluna de equipamento
    equip_col = None
    for col in ["Equipamento", "Cod Recurso"]:
        if col in df.columns:
            equip_col = col
            break

    # Ordenar por equipamento e data (se existir)
    sort_cols = []
    if equip_col:
        sort_cols.append(equip_col)
    date_col = next((col for col in df.columns if "Data" in col), None)
    if date_col:
        sort_cols.append(date_col)

    if sort_cols:
        df = df.sort_values(sort_cols)

    # Calcular acumulados
    for col in qty_cols:
        if df[col].dtype in ['int64', 'float64']:
            new_col = f"{col}_Acumulado"
            if equip_col:
                df[new_col] = df.groupby(equip_col)[col].cumsum()
            else:
                df[new_col] = df[col].cumsum()
            print(f"  ✓ Criada variável acumulativa: {new_col}")

    return df


def apply_one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding em variáveis categóricas.

    Colunas codificadas:
    - Cod Produto
    - Equipamento / Cod Recurso
    - Descrição da massa

    Args:
        df: DataFrame de entrada

    Returns:
        DataFrame com encoding aplicado
    """
    categorical_cols = [
        "Cod Produto",
        "Equipamento",
        "Cod Recurso",
        "Descrição da massa (Composto)",
        "Cód. Un."
    ]

    cols_to_encode = [col for col in categorical_cols if col in df.columns]

    if not cols_to_encode:
        print("  ⚠ Nenhuma coluna categórica encontrada para encoding")
        return df

    for col in cols_to_encode:
        n_unique = df[col].nunique()
        if n_unique <= 50:  # Limite para evitar explosão de dimensionalidade
            df = pd.get_dummies(df, columns=[col], prefix=col.replace(" ", "_").replace(".", ""))
            print(f"  ✓ One-Hot Encoding aplicado: {col} ({n_unique} categorias)")
        else:
            print(f"  ⚠ {col} tem muitas categorias ({n_unique}), pulando encoding")

    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza nomes das colunas.

    Args:
        df: DataFrame de entrada

    Returns:
        DataFrame com colunas renomeadas
    """
    # Remover caracteres especiais e espaços
    df.columns = [
        col.replace(" ", "_")
           .replace(".", "")
           .replace("(", "")
           .replace(")", "")
           .replace("/", "_")
           .replace("-", "_")
        for col in df.columns
    ]

    return df


def main(inicio=None, fim=None, **kwargs) -> dict:
    """
    Função principal - Etapa 2: Pré-processamento e Limpeza.

    Args:
        inicio: Data de início do período (passado pelo pipeline)
        fim: Data de fim do período (passado pelo pipeline)

    Returns:
        Dicionário com resultados da execução
    """
    global _MAINTENANCE_CACHE
    _MAINTENANCE_CACHE = None  # Limpar cache para recarregar dados de manutenção

    print("=" * 60)
    print("ETAPA 2: PRÉ-PROCESSAMENTO E LIMPEZA")
    print("(Conforme fluxos.drawio)")
    print("=" * 60)

    # Verificar arquivo de entrada
    input_file = Path("data_raw.csv")
    if not input_file.exists():
        print(f"\n✗ Arquivo não encontrado: {input_file}")
        print("Execute a Etapa 1 primeiro (s01_data_collection.py)")
        return {"status": "error", "message": "Input file not found"}

    # Carregar dados
    print("\n[1/6] Carregando dados brutos...")
    df = load_raw_data(str(input_file))
    initial_shape = df.shape

    # Etapa de Higienização
    print("\n" + "-" * 40)
    print("HIGIENIZAÇÃO E TRANSFORMAÇÃO")
    print("-" * 40)

    print("\n[2/6] Removendo duplicadas...")
    df = remove_duplicates(df)

    print("\n[3/6] Tratando valores nulos...")
    df = handle_null_values(df)

    print("\n[4/6] Convertendo datas...")
    df = convert_dates(df)

    print("\n[4.1] Calculando dias até manutenção...")
    df = calculate_maintenance_days(df)

    # Etapa de Engenharia de Features
    print("\n" + "-" * 40)
    print("ENGENHARIA DE FEATURES")
    print("-" * 40)

    print("\n[5/7] Gerando variáveis acumulativas...")
    df = generate_cumulative_variables(df)

    print("\n[6/7] Adicionando medições de desgaste...")
    df = add_measurement_features(df)

    print("\n[7/8] Aplicando One-Hot Encoding...")
    df = apply_one_hot_encoding(df)

    # Limpar nomes de colunas
    df = clean_column_names(df)

    # Calcular estatísticas por equipamento (antes de limpar nomes)
    print("\n[8/8] Calculando estatísticas por equipamento...")
    # Recarregar dados sem encoding para estatísticas legíveis
    df_stats = load_raw_data(str(input_file))
    df_stats = remove_duplicates(df_stats)
    df_stats = handle_null_values(df_stats)
    df_stats = convert_dates(df_stats)
    df_stats = calculate_maintenance_days(df_stats)
    df_stats = generate_cumulative_variables(df_stats)
    df_stats = add_measurement_features(df_stats)

    equipment_stats = calculate_equipment_statistics(df_stats)
    if not equipment_stats.empty:
        export_equipment_statistics(equipment_stats, "equipment_stats.csv")

    # Salvar dados preprocessados
    output_file = Path("data_preprocessed.csv")
    df.to_csv(output_file, index=False)

    # Resumo
    final_shape = df.shape
    print("\n" + "=" * 60)
    print("ETAPA 2 CONCLUÍDA")
    print("=" * 60)
    print(f"\nTransformação: {initial_shape} → {final_shape}")
    print(f"Arquivo salvo: {output_file}")

    # Listar colunas
    print(f"\nColunas ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    results = {
        "status": "success",
        "input_shape": initial_shape,
        "output_shape": final_shape,
        "output_file": str(output_file),
        "columns": list(df.columns),
        "has_target": "Manutencao" in df.columns,
        "equipment_stats_file": "equipment_stats.csv" if not equipment_stats.empty else None,
        "num_equipments": len(equipment_stats) if not equipment_stats.empty else 0,
    }

    return results


if __name__ == "__main__":
    main()
