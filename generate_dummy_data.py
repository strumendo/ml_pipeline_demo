"""
Generate dummy data files for the DemoML Pipeline Demo.
This creates synthetic CSV/XLSX files to replace real production data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_MAINT = BASE_DIR / "data" / "manutencao"

# 10 dummy equipment IDs
EQUIPMENT_IDS = [f"EQ-{i:03d}" for i in [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]]

PRODUCTS = [f"PR{random.randint(10000, 99999)}" for _ in range(20)]
COMPOUNDS = ["C-100/A", "C-200/B", "C-300/C", "C-150/D", "C-250/E"]

START_DATE = datetime(2023, 3, 1)
END_DATE = datetime(2025, 12, 31)


def generate_equipment_csv(equip_id: str, n_records: int = None):
    """Generate a production data CSV for one equipment."""
    if n_records is None:
        n_records = random.randint(200, 600)

    dates = pd.date_range(START_DATE, END_DATE, periods=n_records)

    df = pd.DataFrame({
        "Data de Produção": dates.strftime("%Y-%m-%d"),
        "Cód. Ordem": np.random.randint(1000000, 9999999, size=n_records),
        "Cód. Recurso": equip_id,
        "Cód. Produto": np.random.choice(PRODUCTS, size=n_records),
        "Qtd. Produzida": np.random.randint(100, 5000, size=n_records),
        "Qtd. Refugada": np.random.randint(0, 80, size=n_records),
        "Qtd. Retrabalhada": np.random.randint(0, 40, size=n_records),
        "Fator Un.": np.ones(n_records),
        "Cód. Un.": "PC",
        "Descrição da massa (Composto)": np.random.choice(COMPOUNDS, size=n_records),
        "Consumo de massa no item em (Kg/100pçs)": np.round(np.random.uniform(0.3, 2.5, size=n_records), 3),
    })

    csv_path = DATA_RAW / f"{equip_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Created: {csv_path.name} ({n_records} records)")


def generate_consolidated_production():
    """Generate a consolidated production file."""
    all_records = []
    for equip_id in EQUIPMENT_IDS[:5]:
        n = random.randint(50, 150)
        dates = pd.date_range(datetime(2025, 11, 1), datetime(2025, 12, 31), periods=n)
        df = pd.DataFrame({
            "Data de Produção": dates.strftime("%Y-%m-%d"),
            "Cód. Ordem": np.random.randint(1000000, 9999999, size=n),
            "Cód. Recurso": equip_id,
            "Cód. Produto": np.random.choice(PRODUCTS, size=n),
            "Qtd. Produzida": np.random.randint(100, 5000, size=n),
            "Qtd. Refugada": np.random.randint(0, 80, size=n),
            "Qtd. Retrabalhada": np.random.randint(0, 40, size=n),
            "Fator Un.": np.ones(n),
            "Cód. Un.": "PC",
            "Descrição da massa (Composto)": np.random.choice(COMPOUNDS, size=n),
            "Consumo de massa no item em (Kg/100pçs)": np.round(np.random.uniform(0.3, 2.5, size=n), 3),
        })
        all_records.append(df)

    df_all = pd.concat(all_records, ignore_index=True)
    csv_path = DATA_RAW / "DadosProducaodemo2025.xlsx"

    try:
        df_all.to_excel(csv_path, index=False)
        print(f"  Created: {csv_path.name} ({len(df_all)} records)")
    except Exception:
        csv_path = DATA_RAW / "DadosProducaodemo2025.csv"
        df_all.to_csv(csv_path, index=False)
        print(f"  Created: {csv_path.name} ({len(df_all)} records)")


def generate_maintenance_data():
    """Generate dummy maintenance XLSX and CSV files."""
    rows = []
    for equip_id in EQUIPMENT_IDS:
        last_maint = START_DATE + timedelta(days=random.randint(300, 800))
        prev_maint = last_maint - timedelta(days=random.randint(200, 500))
        dias_op = (last_maint - prev_maint).days

        # Cylinder measurements (5 points A-E, Max, Min)
        cil_base = 20.0
        cil_values = [round(cil_base + random.uniform(-0.05, 0.15), 3) for _ in range(5)]
        cil_max = max(cil_values)
        cil_min = min(cil_values)

        # Spindle measurements (4 points A-D, Max, Min)
        fuso_base = 20.0
        fuso_values = [round(fuso_base - random.uniform(0, 0.5), 3) for _ in range(4)]
        fuso_max = max(fuso_values)
        fuso_min = min(fuso_values)

        rows.append([
            None,  # Column A (header area)
            equip_id,  # Column B
            last_maint,  # Column C
            prev_maint,  # Column D
            dias_op,  # Column E
            random.choice(["OK", "Desgaste normal", "Monitorar", None]),  # Column F
        ] + cil_values + [cil_max, cil_min] + fuso_values + [fuso_max, fuso_min])

    # Add header rows
    header1 = ["", "Equipamento", "Última Substituição", "Penúltima Substituição",
                "Dias em Operação", "Observações",
                "Cil A", "Cil B", "Cil C", "Cil D", "Cil E", "Cil Máx", "Cil Mín",
                "Fuso A", "Fuso B", "Fuso C", "Fuso D", "Fuso Máx", "Fuso Mín"]
    header2 = [""] * len(header1)

    all_rows = [header1, header2] + rows
    df = pd.DataFrame(all_rows)

    try:
        xlsx_path = DATA_MAINT / "Dados Manut - 10 Equip - Demo.xlsx"
        df.to_excel(xlsx_path, index=False, header=False)
        print(f"  Created: {xlsx_path.name}")
    except Exception as e:
        print(f"  Warning: Could not create XLSX ({e}). Creating CSV instead.")
        csv_path = DATA_MAINT / "dados_manutencao_demo.csv"
        df.to_csv(csv_path, index=False, header=False)
        print(f"  Created: {csv_path.name}")

    # Also create a simple CSV version
    csv_data = []
    for equip_id in EQUIPMENT_IDS:
        last_maint = START_DATE + timedelta(days=random.randint(300, 800))
        csv_data.append({
            "equipamento": equip_id,
            "data_manutencao": last_maint.strftime("%Y-%m-%d"),
            "tipo": random.choice(["Preventiva", "Corretiva"]),
            "descricao": "Manutenção programada demo",
        })

    df_csv = pd.DataFrame(csv_data)
    csv_path = DATA_MAINT / "dados_manutencao.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"  Created: {csv_path.name}")


def main():
    print("=" * 50)
    print("Generating Dummy Data for DemoML Pipeline")
    print("=" * 50)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_MAINT.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Generating equipment production files...")
    for equip_id in EQUIPMENT_IDS:
        generate_equipment_csv(equip_id)

    print("\n[2/3] Generating consolidated production file...")
    generate_consolidated_production()

    print("\n[3/3] Generating maintenance data...")
    generate_maintenance_data()

    print("\n" + "=" * 50)
    print("Done! All dummy data files created.")
    print("=" * 50)


if __name__ == "__main__":
    main()
