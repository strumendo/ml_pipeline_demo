"""
S01 - Coleta e Integração de Dados
==================================
Etapa 1 do Pipeline conforme fluxos.drawio

O QUE FAZ:
- Carrega múltiplos arquivos CSV de equipamentos (EQ-101.csv, EQ-102.csv, etc.)
- Converte arquivos XLSX para CSV automaticamente se necessário
- Concatena todos em um DataFrame único
- Padroniza nomes de colunas

FLUXO (fluxos.drawio):
CSV Files (EQ-101, EQ-102, EQ-103, ...) → DataFrame Único

ENTRADA:
- Arquivos CSV/XLSX em ml_pipeline_demo/data/raw/

SAÍDA:
- ml_pipeline_demo/outputs/data_raw.csv: DataFrame único com todos os dados brutos
"""

import pandas as pd
from pathlib import Path
import sys

# Adicionar config ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
try:
    from paths import DATA_RAW_DIR, DATA_RAW_FILE, OUTPUTS_DIR
except ImportError:
    # Fallback para caminhos relativos
    DATA_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
    OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
    DATA_RAW_FILE = OUTPUTS_DIR / "data_raw.csv"


def convert_xlsx_to_csv(filepath: Path, force_refresh: bool = True) -> Path:
    """
    Converte arquivo XLSX para CSV.

    Args:
        filepath: Caminho base do arquivo (sem extensão)
        force_refresh: Se True, sempre re-lê o XLSX (ignora CSV existente)

    Returns:
        Caminho do arquivo CSV
    """
    # Usar parent + stem para evitar bug com nomes que contêm pontos
    # Ex: Path("EQ-138.2").with_suffix(".csv") daria "EQ-138.csv" (errado!)
    base_name = filepath.name  # Ex: "EQ-138.2" ou "EQ-101"
    parent = filepath.parent
    csv_path = parent / f"{base_name}.csv"
    xlsx_path = parent / f"{base_name}.xlsx"

    # Se force_refresh=False e CSV existe, usar o CSV
    if not force_refresh and csv_path.exists():
        return csv_path

    # Sempre ler do XLSX para garantir dados completos
    if xlsx_path.exists():
        print(f"  Lendo {xlsx_path.name}...")
        try:
            df = pd.read_excel(xlsx_path)
            # Converter coluna de data para datetime e salvar em formato ISO
            if "Data de Produção" in df.columns:
                # Tentar converter - Excel geralmente já tem datetime
                df["Data de Produção"] = pd.to_datetime(
                    df["Data de Produção"], dayfirst=True, errors='coerce'
                )
            df.to_csv(csv_path, index=False, date_format='%Y-%m-%d')
            return csv_path
        except Exception as e:
            print(f"  Erro na conversão: {e}")
            # Fallback para CSV se existir
            if csv_path.exists():
                return csv_path
            return None

    # Se não tem XLSX mas tem CSV, usar o CSV
    if csv_path.exists():
        return csv_path

    return None


def find_data_directory() -> Path:
    """
    Procura o diretório de dados automaticamente.

    Returns:
        Path do diretório de dados ou None
    """
    # Primeiro, verificar o diretório configurado
    if DATA_RAW_DIR.exists():
        xlsx_files = list(DATA_RAW_DIR.glob("EQ-*.xlsx"))
        csv_files = list(DATA_RAW_DIR.glob("EQ-*.csv"))
        if xlsx_files or csv_files:
            return DATA_RAW_DIR

    # Fallback para outros diretórios possíveis
    possible_dirs = [
        Path("../data/raw"),
        Path("./data/raw"),
        Path("."),
    ]

    for d in possible_dirs:
        if d.exists():
            xlsx_files = list(d.glob("EQ-*.xlsx"))
            csv_files = list(d.glob("EQ-*.csv"))
            if xlsx_files or csv_files:
                return d.resolve()

    return None


def normalize_extended_format(df: pd.DataFrame, equip_name: str) -> pd.DataFrame:
    """
    Normaliza arquivos com formato estendido (como EQ-138.2.xlsx) para o formato padrão.

    Args:
        df: DataFrame com formato estendido
        equip_name: Nome do equipamento

    Returns:
        DataFrame normalizado
    """
    # Colunas padrão esperadas
    standard_cols = [
        "Data de Produção", "Cód. Ordem", "Cód. Recurso", "Cód. Produto",
        "Qtd. Produzida", "Qtd. Refugada", "Qtd. Retrabalhada",
        "Fator Un.", "Cód. Un.", "Descrição da massa (Composto)",
        "Consumo de massa no item em (Kg/100pçs)"
    ]

    # Mapeamento de colunas do formato estendido para o padrão
    column_mapping = {
        "Data de Produção": "Data de Produção",
        "Cód. Ordem": "Cód. Ordem",
        "Cód. Recurso": "Cód. Recurso",
        "Cód. Produto": "Cód. Produto",
        "Qtd. Produzida": "Qtd. Produzida",
        "Qtd. Refugada": "Qtd. Refugada",
        "Qtd. Retrabalhada": "Qtd. Retrabalhada",
        "Fator Un.": "Fator Un.",
        "Cód. Un.": "Cód. Un.",
    }

    # Criar DataFrame normalizado
    normalized = pd.DataFrame()

    for std_col, src_col in column_mapping.items():
        if src_col in df.columns:
            normalized[std_col] = df[src_col]
        else:
            normalized[std_col] = None

    # Colunas que podem não existir no formato estendido
    if "Descrição da massa (Composto)" not in normalized.columns or normalized["Descrição da massa (Composto)"].isna().all():
        # Tentar usar "Descrição" ou "Nome. Produto"
        if "Descrição" in df.columns:
            normalized["Descrição da massa (Composto)"] = df["Descrição"]
        elif "Nome. Produto" in df.columns:
            normalized["Descrição da massa (Composto)"] = df["Nome. Produto"]
        else:
            normalized["Descrição da massa (Composto)"] = "N/A"

    if "Consumo de massa no item em (Kg/100pçs)" not in normalized.columns:
        normalized["Consumo de massa no item em (Kg/100pçs)"] = 0.0

    return normalized


def load_extended_xlsx_files(data_dir: Path) -> list:
    """
    Carrega arquivos XLSX com formato estendido (ex: EQ-138.2.xlsx).

    Args:
        data_dir: Diretório de dados

    Returns:
        Lista de tuplas (nome_equipamento, DataFrame)
    """
    dataframes = []

    # Procurar arquivos com padrão diferente (ex: EQ-XXX.2.xlsx)
    extended_files = list(data_dir.glob("EQ-*.*.xlsx"))

    for filepath in extended_files:
        equip_name = filepath.stem.split(".")[0]  # Pegar apenas "EQ-138" de "EQ-138.2"

        try:
            print(f"  Carregando arquivo estendido: {filepath.name}")
            df = pd.read_excel(filepath)

            # Normalizar para formato padrão
            df_normalized = normalize_extended_format(df, equip_name)
            df_normalized["Equipamento"] = equip_name

            # Converter coluna de data para datetime
            if "Data de Produção" in df_normalized.columns:
                df_normalized["Data de Produção"] = pd.to_datetime(
                    df_normalized["Data de Produção"], dayfirst=True, errors='coerce'
                )

            # Adicionar coluna de origem para identificação
            df_normalized["Fonte_Dados"] = filepath.name

            dataframes.append((f"{equip_name}_ext", df_normalized))
            print(f"  ✓ {filepath.name}: {len(df_normalized)} registros (dados históricos)")

        except Exception as e:
            print(f"  ✗ {filepath.name}: Erro ao carregar - {e}")

    return dataframes


def load_dados_producao_files(data_dir: Path) -> list:
    """
    Carrega arquivos DadosProducao*.xlsx com dados de produção consolidados.

    Args:
        data_dir: Diretório de dados

    Returns:
        Lista de tuplas (nome_arquivo, DataFrame)
    """
    dataframes = []

    # Procurar arquivos DadosProducao*.xlsx
    producao_files = list(data_dir.glob("DadosProducao*.xlsx"))

    for filepath in producao_files:
        try:
            print(f"  Carregando arquivo de produção: {filepath.name}")
            df = pd.read_excel(filepath)

            # Converter coluna de data para datetime
            if "Data de Produção" in df.columns:
                df["Data de Produção"] = pd.to_datetime(
                    df["Data de Produção"], dayfirst=True, errors='coerce'
                )

            # Extrair equipamento da coluna Cód. Recurso
            if "Cód. Recurso" in df.columns:
                df["Equipamento"] = df["Cód. Recurso"]

            # Adicionar coluna de origem para identificação
            df["Fonte_Dados"] = filepath.name

            dataframes.append((filepath.stem, df))
            print(f"  ✓ {filepath.name}: {len(df)} registros (dados de produção)")

        except Exception as e:
            print(f"  ✗ {filepath.name}: Erro ao carregar - {e}")

    return dataframes


def load_csv_files(data_dir: Path, pattern: str = "EQ-*.csv") -> list:
    """
    Carrega todos os arquivos CSV que correspondem ao padrão.
    Exclui arquivos com formato estendido (ex: EQ-138.2) que são
    tratados por load_extended_xlsx_files.

    Args:
        data_dir: Diretório de dados
        pattern: Padrão glob para encontrar arquivos

    Returns:
        Lista de tuplas (nome_equipamento, DataFrame)
    """
    dataframes = []

    # Procurar arquivos CSV e XLSX
    csv_files = sorted(data_dir.glob(pattern))
    xlsx_files = sorted(data_dir.glob(pattern.replace(".csv", ".xlsx")))

    # Identificar arquivos com formato estendido (ex: EQ-138.2.xlsx)
    # Esses são tratados por load_extended_xlsx_files, não carregar aqui
    extended_stems = set()
    for f in data_dir.glob("EQ-*.*.xlsx"):
        extended_stems.add(f.stem)  # Ex: "EQ-138.2"

    all_files = set()
    for f in csv_files:
        all_files.add(f.stem)
    for f in xlsx_files:
        all_files.add(f.stem)

    # Remover arquivos estendidos
    all_files -= extended_stems
    if extended_stems:
        print(f"  (Excluindo {len(extended_stems)} arquivo(s) estendido(s): {sorted(extended_stems)})")

    for equip_name in sorted(all_files):
        filepath = convert_xlsx_to_csv(data_dir / equip_name)

        if filepath and filepath.exists():
            try:
                df = pd.read_csv(filepath)
                df["Equipamento"] = equip_name
                dataframes.append((equip_name, df))
                print(f"  ✓ {equip_name}: {len(df)} registros")
            except Exception as e:
                print(f"  ✗ {equip_name}: Erro ao carregar - {e}")

    return dataframes


def merge_dataframes(dataframes: list) -> pd.DataFrame:
    """
    Concatena todos os DataFrames em um único DataFrame.
    Remove registros duplicados (mesma ordem, data, equipamento, produto e quantidade).

    Args:
        dataframes: Lista de tuplas (nome, DataFrame)

    Returns:
        DataFrame consolidado e deduplicado
    """
    if not dataframes:
        return pd.DataFrame()

    dfs = [df for _, df in dataframes]
    merged = pd.concat(dfs, ignore_index=True)

    # Remover duplicatas exatas por colunas-chave
    dedup_cols = ["Cód. Ordem", "Data de Produção", "Cód. Recurso", "Cód. Produto", "Qtd. Produzida"]
    available_cols = [c for c in dedup_cols if c in merged.columns]
    if available_cols:
        antes = len(merged)
        merged = merged.drop_duplicates(subset=available_cols, keep="first")
        removidos = antes - len(merged)
        if removidos > 0:
            print(f"  ⚠ Removidos {removidos} registros duplicados ({antes} → {len(merged)})")

    return merged


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Gera dados sintéticos quando não há dados reais disponíveis.

    Args:
        n_samples: Número de amostras a gerar

    Returns:
        DataFrame com dados sintéticos
    """
    import numpy as np

    equipamentos = ["EQ-101", "EQ-102", "EQ-103", "EQ-104"]
    produtos = ["PR10001", "PR10002", "PR10003", "PR10004"]

    df = pd.DataFrame({
        "Data de Produção": pd.to_datetime(
            np.random.choice(pd.date_range("2023-01-01", "2024-12-31"), size=n_samples)
        ).strftime("%d/%m/%Y"),
        "Cód. Ordem": np.random.randint(1000000, 9999999, size=n_samples),
        "Equipamento": np.random.choice(equipamentos, size=n_samples),
        "Cod Produto": np.random.choice(produtos, size=n_samples),
        "Qtd. Produzida": np.random.randint(100, 5000, size=n_samples),
        "Qtd. Refugada": np.random.randint(0, 100, size=n_samples),
        "Qtd. Retrabalhada": np.random.randint(0, 50, size=n_samples),
        "Fator Un.": np.ones(n_samples),
        "Cód. Un.": np.full(n_samples, "PC"),
        "Descrição da massa (Composto)": np.random.choice(["C-100/A", "C-200/B", "C-300/C"], size=n_samples),
        "Consumo de massa no item em (Kg/100pçs)": np.random.uniform(0.5, 2.0, size=n_samples).round(3),
    })

    print(f"  ✓ Dados sintéticos gerados: {n_samples} registros")

    return df


def main(inicio=None, fim=None, **kwargs) -> dict:
    """
    Função principal - Etapa 1: Coleta e Integração.

    Args:
        inicio: Data de início do período (YYYY-MM-DD), opcional
        fim: Data de fim do período (YYYY-MM-DD), opcional

    Returns:
        Dicionário com resultados da execução
    """
    print("=" * 60)
    print("ETAPA 1: COLETA E INTEGRAÇÃO DE DADOS")
    print("(Conforme fluxos.drawio)")
    print("=" * 60)

    # Procurar diretório de dados
    data_dir = find_data_directory()

    if data_dir:
        print(f"\nDiretório de dados: {data_dir}")

        # Carregar arquivos padrão (CSV/XLSX)
        print("\nCarregando arquivos CSV/XLSX padrão...")
        dataframes = load_csv_files(data_dir)

        # Carregar arquivos com formato estendido (dados históricos)
        print("\nCarregando arquivos com formato estendido (dados históricos)...")
        extended_dataframes = load_extended_xlsx_files(data_dir)

        # Carregar arquivos DadosProducao*.xlsx
        print("\nCarregando arquivos DadosProducao (dados consolidados)...")
        producao_dataframes = load_dados_producao_files(data_dir)

        # Combinar todos os dataframes
        all_dataframes = dataframes + extended_dataframes + producao_dataframes

        if all_dataframes:
            # Mesclar DataFrames
            print("\nIntegrando em DataFrame único...")
            df_merged = merge_dataframes(all_dataframes)

            # Converter e normalizar datas
            if "Data de Produção" in df_merged.columns:
                try:
                    # Guardar valores originais como string para fallback
                    original_values = df_merged["Data de Produção"].astype(str)

                    # Tentativa 1: Auto-detecção
                    df_merged["Data de Produção"] = pd.to_datetime(
                        df_merged["Data de Produção"], errors="coerce"
                    )

                    # Tentativa 2: Para NaTs restantes, tentar dayfirst=True
                    mask = df_merged["Data de Produção"].isna()
                    if mask.any():
                        df_merged.loc[mask, "Data de Produção"] = pd.to_datetime(
                            original_values[mask], dayfirst=True, errors="coerce"
                        )

                    data_min = df_merged["Data de Produção"].min()
                    data_max = df_merged["Data de Produção"].max()
                    print(f"\n  Período total dos dados: {data_min.strftime('%d/%m/%Y')} a {data_max.strftime('%d/%m/%Y')}")
                    total_dias = (data_max - data_min).days
                    print(f"  Duração total: {total_dias} dias ({total_dias // 30} meses)")
                except Exception as e:
                    print(f"  Aviso: Não foi possível calcular período - {e}")

            # Aplicar filtro de período se informado
            registros_antes = len(df_merged)
            if inicio or fim:
                if "Data de Produção" in df_merged.columns:
                    print(f"\n  Aplicando filtro de período: {inicio or 'início'} a {fim or 'fim'}...")
                    if inicio:
                        dt_inicio = pd.to_datetime(inicio)
                        df_merged = df_merged[df_merged["Data de Produção"] >= dt_inicio]
                    if fim:
                        dt_fim = pd.to_datetime(fim)
                        df_merged = df_merged[df_merged["Data de Produção"] <= dt_fim]

                    registros_filtrados = registros_antes - len(df_merged)
                    print(f"  Registros filtrados: {registros_filtrados} removidos, {len(df_merged)} restantes")

                    if len(df_merged) > 0:
                        data_min = df_merged["Data de Produção"].min()
                        data_max = df_merged["Data de Produção"].max()
                        print(f"  Período filtrado: {data_min.strftime('%d/%m/%Y')} a {data_max.strftime('%d/%m/%Y')}")

            # Salvar
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            df_merged.to_csv(DATA_RAW_FILE, index=False, date_format='%Y-%m-%d')

            print(f"\n✓ DataFrame único salvo: {DATA_RAW_FILE}")
            print(f"  Total: {len(df_merged)} registros de {len(all_dataframes)} fontes de dados")
            if extended_dataframes:
                print(f"  (inclui {len(extended_dataframes)} arquivo(s) com dados históricos)")

            results = {
                "status": "success",
                "source": "real_data",
                "equipamentos": len(all_dataframes),
                "registros": len(df_merged),
                "output_file": str(DATA_RAW_FILE),
                "colunas": list(df_merged.columns),
                "inicio": inicio,
                "fim": fim,
            }
        else:
            print("\n⚠ Nenhum arquivo de equipamento encontrado.")
            print("Gerando dados sintéticos...")

            df_merged = generate_synthetic_data(1000)
            output_path = Path("data_raw.csv")
            df_merged.to_csv(output_path, index=False)

            results = {
                "status": "success",
                "source": "synthetic",
                "registros": len(df_merged),
                "output_file": str(output_path),
            }
    else:
        print("\n⚠ Diretório de dados não encontrado.")
        print("Gerando dados sintéticos para demonstração...")

        df_merged = generate_synthetic_data(1000)
        output_path = Path("data_raw.csv")
        df_merged.to_csv(output_path, index=False)

        results = {
            "status": "success",
            "source": "synthetic",
            "registros": len(df_merged),
            "output_file": str(output_path),
        }

    print("\n" + "=" * 60)
    print("ETAPA 1 CONCLUÍDA")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
