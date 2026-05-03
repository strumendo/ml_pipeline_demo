"""
Gerador de dados sintéticos para o Pipeline DemoML.

Schema fiel aos arquivos de referência em data/_reference_schema/:
- Produção:    11 colunas PT-BR (Data de Produção, Cód. Ordem, Cód. Recurso, ...).
- Manutenção:  19 colunas com header em 2 níveis (Histórico + Componente A 5pts + Componente B 4pts).
- Preventivas: 6 colunas (Equipamento, GrpLisTar., Texto item man., Nº solicitação, Dta.iníc.progr., Ordem).

Saídas:
- data/raw/EQ-XXX.csv               (1 arquivo por equipamento)
- data/raw/DadosProducaodemo2025.csv  (consolidado para 1 período)
- data/manutencao/dados_manutencao.xlsx               (schema de manutenção)
- data/manutencao/historico_preventivas.xlsx          (schema de preventivas)
- data/manutencao/dados_manutencao.csv                (versão CSV simples)
- data/manutencao/dados_manutencao_demo.csv           (versão CSV achatada)
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_MAINT = BASE_DIR / "data" / "manutencao"

EQUIPMENT_IDS = [f"EQ-{i:03d}" for i in range(101, 128)]

COMPOUNDS = [
    "MASSA C315/1", "P250/14", "MASSA C420/191", "C424/50", "FP75614/49",
    "MASSA C420/134", "N148/18", "N-184/2", "N144/83", "N146/9",
    "C424/40 ESCURO", "MASSA C403/32", "N-142/67", "MASSA C420/96",
    "N-142/44", "MASSA C420/95", "C424/64", "P248/40", "MASSA C316/2",
    "EP 4053/1",
]

PRODUCT_CODES = [f"PR{i:05d}" for i in range(10000, 10080)]

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 3, 31)

OBSERVACOES_POOL = [
    "OK", "Substituição prevista", "Desgaste normal",
    "Monitorar próxima medição", None, "Componente novo",
]


def _equipment_records(equip_id: str, n_records: int) -> pd.DataFrame:
    dates = pd.date_range(START_DATE, END_DATE, periods=n_records).strftime("%d/%m/%Y") + " 00:00:00"
    fator_un = np.random.choice([1, 20], size=n_records, p=[0.7, 0.3])
    return pd.DataFrame({
        "Data de Produção": dates,
        "Cód. Ordem": np.random.randint(2_700_000, 2_800_000, size=n_records),
        "Cód. Recurso": equip_id,
        "Cód. Produto": np.random.choice(PRODUCT_CODES, size=n_records),
        "Qtd. Produzida": np.random.randint(40, 5000, size=n_records),
        "Qtd. Refugada": np.random.randint(0, 80, size=n_records),
        "Qtd. Retrabalhada": np.random.randint(0, 40, size=n_records),
        "Fator Un.": fator_un,
        "Cód. Un.": "PC",
        "Descrição da massa (Composto)": np.random.choice(COMPOUNDS, size=n_records),
        "Consumo de massa no item em (Kg/100pçs)": np.round(np.random.uniform(0.18, 2.5, size=n_records), 3),
    })


def generate_equipment_csv(equip_id: str) -> int:
    n_records = random.randint(200, 700)
    df = _equipment_records(equip_id, n_records)
    csv_path = DATA_RAW / f"{equip_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Created: {csv_path.name} ({n_records} records)")
    return n_records


def generate_consolidated_production() -> None:
    frames = []
    for equip_id in EQUIPMENT_IDS[:5]:
        n = random.randint(80, 200)
        df = _equipment_records(equip_id, n)
        df["Data de Produção"] = pd.date_range(
            datetime(2025, 11, 1), datetime(2026, 3, 31), periods=n
        ).strftime("%d/%m/%Y") + " 00:00:00"
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    csv_path = DATA_RAW / "DadosProducaodemo2025.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"  Created: {csv_path.name} ({len(df_all)} records)")


HEADER_LVL1 = [
    None, "Histórico Manutenções (Substituições)", None, None, None, None,
    "Medições Componente A", None, None, None, None, None, None,
    "Medições Componente B", None, None, None, None, None,
]
HEADER_LVL2 = [
    None, "Equipamento", "Data execução da última substituição",
    "Data da penúltima substituição", "Dias em operação", "Observações",
    "A", "B", "C", "D", "E", "Máximo", "Mínimo",
    "A", "B", "C", "D", "Máximo", "Mínimo",
]


def _component_a_values():
    base = 20.0
    vals = [round(base + random.uniform(-0.05, 0.15), 3) for _ in range(5)]
    return vals + [max(vals), min(vals)]


def _component_b_values():
    base = 20.0
    vals = [round(base - random.uniform(0, 0.5), 3) for _ in range(4)]
    return vals + [max(vals), min(vals)]


def generate_maintenance_xlsx() -> None:
    rows = [HEADER_LVL1, HEADER_LVL2]
    flat_rows = []
    for equip_id in EQUIPMENT_IDS:
        last_maint = START_DATE + timedelta(days=random.randint(300, 800))
        prev_maint = last_maint - timedelta(days=random.randint(200, 500))
        dias_op = (last_maint - prev_maint).days
        comp_a = _component_a_values()
        comp_b = _component_b_values()
        body = [
            None,
            equip_id,
            last_maint,
            prev_maint,
            dias_op,
            random.choice(OBSERVACOES_POOL),
            *comp_a,
            *comp_b,
        ]
        rows.append(body)
        flat_rows.append(body)

    df = pd.DataFrame(rows)
    xlsx_path = DATA_MAINT / "dados_manutencao.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Hist. Medições e Manutenções",
                    index=False, header=False)
    print(f"  Created: {xlsx_path.name}")

    flat_cols = [c for c in HEADER_LVL2 if c is not None]
    flat_data = [[v for v, h in zip(r, HEADER_LVL2) if h is not None] for r in flat_rows]
    pd.DataFrame(flat_data, columns=flat_cols).to_csv(
        DATA_MAINT / "dados_manutencao_demo.csv", index=False
    )
    print(f"  Created: dados_manutencao_demo.csv")

    simple = []
    for equip_id in EQUIPMENT_IDS:
        last_maint = START_DATE + timedelta(days=random.randint(300, 800))
        simple.append({
            "equipamento": equip_id,
            "data_manutencao": last_maint.strftime("%Y-%m-%d"),
            "tipo": random.choice(["Preventiva", "Corretiva"]),
            "descricao": "Manutenção programada (demo)",
        })
    pd.DataFrame(simple).to_csv(DATA_MAINT / "dados_manutencao.csv", index=False)
    print(f"  Created: dados_manutencao.csv")


def generate_preventivas_xlsx() -> None:
    rows = []
    for equip_id in EQUIPMENT_IDS:
        n_events = random.randint(3, 6)
        base = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 360))
        for i in range(n_events):
            data_evento = base + timedelta(days=random.randint(360, 540) * (i + 1))
            rows.append({
                "Equipamento": equip_id,
                "GrpLisTar.": "PREV.MEC",
                "Texto item man.": "PREVENTIVA MECÂNICA (COMPLETA)",
                "Nº solicitação": i + 1,
                "Dta.iníc.progr.": data_evento,
                "Ordem": random.randint(5_070_000, 5_110_000) if i < n_events - 1 else "",
            })
    df = pd.DataFrame(rows)
    xlsx_path = DATA_MAINT / "historico_preventivas.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Planilha1", index=False)
    print(f"  Created: {xlsx_path.name} ({len(df)} eventos)")


def main() -> None:
    print("=" * 60)
    print("Gerando dados sintéticos para o Pipeline DemoML")
    print("=" * 60)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_MAINT.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Apontamentos de produção por equipamento ({len(EQUIPMENT_IDS)} equipamentos)...")
    total = 0
    for equip_id in EQUIPMENT_IDS:
        total += generate_equipment_csv(equip_id)
    print(f"  Total: {total} apontamentos")

    print("\n[2/4] Arquivo consolidado de produção...")
    generate_consolidated_production()

    print("\n[3/4] Manutenções por equipamento (schema 2-níveis)...")
    generate_maintenance_xlsx()

    print("\n[4/4] Histórico de preventivas...")
    generate_preventivas_xlsx()

    print("\n" + "=" * 60)
    print("Concluído. Todos os arquivos sintéticos foram criados.")
    print("=" * 60)


if __name__ == "__main__":
    main()
