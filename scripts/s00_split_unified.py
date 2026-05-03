"""
S00 - Separação de Arquivo Único por Equipamento
=================================================
Etapa 0 do Pipeline (opcional)

O QUE FAZ:
- Procura arquivos Excel em data/arquivo_unico/
- Separa cada arquivo por equipamento (coluna "Cód. Recurso")
- Salva .xlsx e .csv em data/raw/ com nome do equipamento
- Se já existir arquivo com mesmo nome, appenda os dados
- Move o arquivo original para data/arquivo_unico_processado/
- Converte todos os .xlsx em data/raw/ para .csv

ENTRADA:
- data/arquivo_unico/*.xlsx

SAÍDA:
- data/raw/<equipamento>.csv e .xlsx (por equipamento)
- data/arquivo_unico_processado/ (originais processados)
"""

import pandas as pd
from pathlib import Path
import sys
import shutil
from datetime import datetime

# Adicionar config ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
try:
    from paths import (
        DATA_RAW_DIR, DATA_ARQUIVO_UNICO_DIR,
        DATA_ARQUIVO_UNICO_PROCESSADO_DIR
    )
except ImportError:
    BASE_DIR = Path(__file__).parent.parent
    DATA_RAW_DIR = BASE_DIR / "data" / "raw"
    DATA_ARQUIVO_UNICO_DIR = BASE_DIR / "data" / "arquivo_unico"
    DATA_ARQUIVO_UNICO_PROCESSADO_DIR = BASE_DIR / "data" / "arquivo_unico_processado"


def find_unified_files() -> list:
    """
    Procura arquivos Excel na pasta data/arquivo_unico/.

    Returns:
        Lista de Paths dos arquivos encontrados
    """
    if not DATA_ARQUIVO_UNICO_DIR.exists():
        return []

    files = sorted(DATA_ARQUIVO_UNICO_DIR.glob("*.xlsx"))
    return files


def split_by_equipment(filepath: Path) -> dict:
    """
    Separa um arquivo Excel por equipamento (coluna "Cód. Recurso").
    Salva cada equipamento como .xlsx e .csv em data/raw/.
    Se já existir, appenda os dados.

    Args:
        filepath: Caminho do arquivo Excel

    Returns:
        Dicionário com resumo do processamento
    """
    print(f"\n  Processando: {filepath.name}")

    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"  Erro ao ler {filepath.name}: {e}")
        return {"status": "error", "message": str(e)}

    # Identificar coluna de equipamento
    equip_col = None
    for col_name in ["Cód. Recurso", "Cod. Recurso", "Cod Recurso", "Equipamento"]:
        if col_name in df.columns:
            equip_col = col_name
            break

    if equip_col is None:
        print(f"  Coluna de equipamento não encontrada em {filepath.name}")
        print(f"  Colunas disponíveis: {list(df.columns)}")
        return {"status": "error", "message": "Equipment column not found"}

    # Converter datas se existir coluna
    if "Data de Produção" in df.columns:
        df["Data de Produção"] = pd.to_datetime(
            df["Data de Produção"], dayfirst=True, errors='coerce'
        )

    equipments = df[equip_col].dropna().unique()
    print(f"  Equipamentos encontrados: {len(equipments)}")

    results = {"equipments": [], "total_records": 0}

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    for equip_name in sorted(equipments):
        equip_name_str = str(equip_name).strip()
        if not equip_name_str:
            continue

        group_df = df[df[equip_col] == equip_name].copy()
        csv_path = DATA_RAW_DIR / f"{equip_name_str}.csv"
        xlsx_path = DATA_RAW_DIR / f"{equip_name_str}.xlsx"

        # Se já existir, appenda
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                # Converter datas no existente
                if "Data de Produção" in existing_df.columns:
                    existing_df["Data de Produção"] = pd.to_datetime(
                        existing_df["Data de Produção"], errors='coerce'
                    )
                combined = pd.concat([existing_df, group_df], ignore_index=True)
                combined = combined.drop_duplicates()
                n_new = len(combined) - len(existing_df)
                print(f"    {equip_name_str}: {len(group_df)} registros ({n_new} novos, {len(combined)} total)")
            except Exception:
                combined = group_df
                print(f"    {equip_name_str}: {len(group_df)} registros (novo arquivo)")
        else:
            combined = group_df
            print(f"    {equip_name_str}: {len(group_df)} registros (novo arquivo)")

        # Salvar CSV e XLSX
        combined.to_csv(csv_path, index=False, date_format='%Y-%m-%d')
        combined.to_excel(xlsx_path, index=False)

        results["equipments"].append(equip_name_str)
        results["total_records"] += len(combined)

    results["status"] = "success"
    return results


def move_to_processed(filepath: Path):
    """
    Move arquivo processado para data/arquivo_unico_processado/.
    Se já existir arquivo com mesmo nome, adiciona timestamp.

    Args:
        filepath: Caminho do arquivo a mover
    """
    DATA_ARQUIVO_UNICO_PROCESSADO_DIR.mkdir(parents=True, exist_ok=True)

    dest = DATA_ARQUIVO_UNICO_PROCESSADO_DIR / filepath.name

    if dest.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = filepath.stem
        suffix = filepath.suffix
        dest = DATA_ARQUIVO_UNICO_PROCESSADO_DIR / f"{stem}_{timestamp}{suffix}"

    shutil.move(str(filepath), str(dest))
    print(f"  Movido para: {dest.name}")


def convert_all_xlsx_to_csv(raw_dir: Path = None):
    """
    Converte todos os .xlsx em data/raw/ para .csv (mantendo ambos).

    Args:
        raw_dir: Diretório raw (padrão: DATA_RAW_DIR)
    """
    if raw_dir is None:
        raw_dir = DATA_RAW_DIR

    if not raw_dir.exists():
        return

    xlsx_files = sorted(raw_dir.glob("EQ-*.xlsx"))
    converted = 0

    for xlsx_path in xlsx_files:
        csv_path = xlsx_path.with_suffix(".csv")
        if not csv_path.exists():
            try:
                df = pd.read_excel(xlsx_path)
                if "Data de Produção" in df.columns:
                    df["Data de Produção"] = pd.to_datetime(
                        df["Data de Produção"], dayfirst=True, errors='coerce'
                    )
                df.to_csv(csv_path, index=False, date_format='%Y-%m-%d')
                converted += 1
                print(f"    Convertido: {xlsx_path.name} -> {csv_path.name}")
            except Exception as e:
                print(f"    Erro ao converter {xlsx_path.name}: {e}")

    if converted > 0:
        print(f"  Total convertidos: {converted} arquivos")
    else:
        print(f"  Todos os .xlsx já possuem versão .csv")


def main(**kwargs) -> dict:
    """
    Função principal - Etapa 0: Separação de Arquivo Único.

    Returns:
        Dicionário com resultados da execução
    """
    print("=" * 60)
    print("ETAPA 0: SEPARAÇÃO DE ARQUIVO ÚNICO POR EQUIPAMENTO")
    print("=" * 60)

    # Procurar arquivos na pasta arquivo_unico
    print("\n[1/3] Procurando arquivos em data/arquivo_unico/...")
    unified_files = find_unified_files()

    if not unified_files:
        print("  Nenhum arquivo encontrado em data/arquivo_unico/")
        print("  Verificando conversão de .xlsx existentes em data/raw/...")

        # Mesmo sem arquivos novos, garantir conversão xlsx->csv
        print("\n[2/3] Verificando conversão XLSX -> CSV em data/raw/...")
        convert_all_xlsx_to_csv()

        print("\n" + "=" * 60)
        print("ETAPA 0 CONCLUÍDA (sem arquivos novos)")
        print("=" * 60)

        return {
            "status": "success",
            "unified_files": 0,
            "message": "No unified files to process"
        }

    print(f"  Encontrados: {len(unified_files)} arquivo(s)")
    for f in unified_files:
        print(f"    - {f.name}")

    # Processar cada arquivo
    print("\n[2/3] Separando por equipamento...")
    all_results = []
    for filepath in unified_files:
        result = split_by_equipment(filepath)
        all_results.append(result)

        if result.get("status") == "success":
            move_to_processed(filepath)

    # Converter xlsx -> csv para todos em data/raw/
    print("\n[3/3] Verificando conversão XLSX -> CSV em data/raw/...")
    convert_all_xlsx_to_csv()

    # Resumo
    total_equip = sum(len(r.get("equipments", [])) for r in all_results)
    total_records = sum(r.get("total_records", 0) for r in all_results)

    print("\n" + "=" * 60)
    print("ETAPA 0 CONCLUÍDA")
    print("=" * 60)
    print(f"\nArquivos processados: {len(unified_files)}")
    print(f"Equipamentos separados: {total_equip}")
    print(f"Total de registros: {total_records}")

    return {
        "status": "success",
        "unified_files": len(unified_files),
        "equipments": total_equip,
        "total_records": total_records,
    }


if __name__ == "__main__":
    main()
