"""
S07 - Cruzamentos Auxiliares (histórico × produção)
====================================================
Etapa 7 do Pipeline (opcional)

O QUE FAZ:
- Lê dados brutos de produção (data/raw/EQ-*.csv) e manutenção (data/manutencao/)
- Gera fotografias temporais e janelas de operação por equipamento:
    - historico_completo.csv  : todas as leituras de manutenção (várias por equipamento se múltiplas fontes)
    - historico_recente.csv   : última leitura por equipamento
    - janelas_operacao.csv    : janelas entre substituições consecutivas (com dias_operacao + produção total na janela)
    - ociosidade.csv          : dias sem produção entre a última troca e hoje
- Saídas alimentam o estágio 8 (prescrição)

ENTRADAS:
- data/raw/EQ-*.csv (apontamentos)
- data/manutencao/dados_manutencao.xlsx (substituições + medições)
- data/manutencao/historico_preventivas.xlsx (eventos preventivos)

SAÍDAS:
- outputs/historico_completo.csv
- outputs/historico_recente.csv
- outputs/janelas_operacao.csv
- outputs/ociosidade.csv
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR / "config"))

try:
    from paths import (
        DATA_RAW_DIR, DATA_MANUTENCAO_DIR,
        HISTORICO_COMPLETO_FILE, HISTORICO_RECENTE_FILE,
        JANELAS_OPERACAO_FILE, OCIOSIDADE_FILE,
    )
except ImportError:
    DATA_RAW_DIR = BASE_DIR / "data" / "raw"
    DATA_MANUTENCAO_DIR = BASE_DIR / "data" / "manutencao"
    HISTORICO_COMPLETO_FILE = Path("historico_completo.csv")
    HISTORICO_RECENTE_FILE = Path("historico_recente.csv")
    JANELAS_OPERACAO_FILE = Path("janelas_operacao.csv")
    OCIOSIDADE_FILE = Path("ociosidade.csv")


def _read_maintenance_full() -> pd.DataFrame:
    """Lê data/manutencao/dados_manutencao.xlsx (header de 2 níveis)."""
    files = list(DATA_MANUTENCAO_DIR.glob("dados_manutencao*.xlsx")) + \
            list(DATA_MANUTENCAO_DIR.glob("Dados Manut*.xlsx"))
    if not files:
        return pd.DataFrame()
    f = max(files, key=lambda p: p.stat().st_mtime)

    df = pd.read_excel(f, header=None)
    rows = []
    for idx, row in df.iterrows():
        if idx < 2:
            continue
        equip = row[1]
        if pd.isna(equip) or not str(equip).startswith("EQ-"):
            continue
        rows.append({
            "equipamento": str(equip).strip(),
            "data_ultima_substituicao": pd.to_datetime(row[2], errors="coerce"),
            "data_penultima_substituicao": pd.to_datetime(row[3], errors="coerce"),
            "dias_em_operacao": int(row[4]) if pd.notna(row[4]) else None,
            "observacoes": str(row[5]) if pd.notna(row[5]) else None,
            "componente_a_max": _safe_float(row[11]),
            "componente_a_min": _safe_float(row[12]),
            "componente_b_max": _safe_float(row[17]),
            "componente_b_min": _safe_float(row[18]),
            "arquivo_origem": f.name,
        })
    return pd.DataFrame(rows)


def _read_preventivas() -> pd.DataFrame:
    """Lê data/manutencao/historico_preventivas.xlsx."""
    files = list(DATA_MANUTENCAO_DIR.glob("historico_preventivas*.xlsx")) + \
            list(DATA_MANUTENCAO_DIR.glob("Histórico Geral Preventivas*.xlsx"))
    if not files:
        return pd.DataFrame()
    f = max(files, key=lambda p: p.stat().st_mtime)
    df = pd.read_excel(f)
    df.columns = [c.strip() for c in df.columns]

    rename = {
        "Equipamento": "equipamento",
        "Dta.iníc.progr.": "data_evento",
        "Texto item man.": "tipo_evento",
        "Nº solicitação": "n_solicitacao",
        "Ordem": "ordem",
    }
    df = df.rename(columns=rename)
    df["equipamento"] = df["equipamento"].astype(str).str.strip()
    df["data_evento"] = pd.to_datetime(df["data_evento"], errors="coerce")
    df = df[df["equipamento"].str.startswith("EQ-")]
    df["arquivo_origem"] = f.name
    return df[["equipamento", "data_evento", "tipo_evento", "n_solicitacao", "ordem", "arquivo_origem"]]


def _read_production() -> pd.DataFrame:
    """Concatena todos os EQ-*.csv em data/raw/."""
    files = sorted(DATA_RAW_DIR.glob("EQ-*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["arquivo_origem"] = f.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if "Data de Produção" in df.columns:
        df["data_producao"] = pd.to_datetime(df["Data de Produção"], dayfirst=True, errors="coerce")
    if "Cód. Recurso" in df.columns:
        df["equipamento"] = df["Cód. Recurso"].astype(str).str.strip()
    return df


def _safe_float(v):
    try:
        return float(v) if pd.notna(v) else None
    except (ValueError, TypeError):
        return None


def build_historico_completo(maint: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """Une todas as leituras (substituições + preventivas) em um único histórico longo."""
    rows = []
    for _, r in maint.iterrows():
        if pd.notna(r["data_ultima_substituicao"]):
            rows.append({
                "equipamento": r["equipamento"],
                "data_evento": r["data_ultima_substituicao"],
                "tipo_evento": "ULTIMA_SUBSTITUICAO",
                "dias_em_operacao": r["dias_em_operacao"],
                "componente_a_max": r["componente_a_max"],
                "componente_a_min": r["componente_a_min"],
                "componente_b_max": r["componente_b_max"],
                "componente_b_min": r["componente_b_min"],
                "observacoes": r["observacoes"],
                "arquivo_origem": r["arquivo_origem"],
            })
        if pd.notna(r["data_penultima_substituicao"]):
            rows.append({
                "equipamento": r["equipamento"],
                "data_evento": r["data_penultima_substituicao"],
                "tipo_evento": "PENULTIMA_SUBSTITUICAO",
                "dias_em_operacao": None,
                "componente_a_max": None,
                "componente_a_min": None,
                "componente_b_max": None,
                "componente_b_min": None,
                "observacoes": None,
                "arquivo_origem": r["arquivo_origem"],
            })

    for _, r in prev.iterrows():
        rows.append({
            "equipamento": r["equipamento"],
            "data_evento": r["data_evento"],
            "tipo_evento": "PREVENTIVA",
            "dias_em_operacao": None,
            "componente_a_max": None,
            "componente_a_min": None,
            "componente_b_max": None,
            "componente_b_min": None,
            "observacoes": str(r.get("tipo_evento") or ""),
            "arquivo_origem": r["arquivo_origem"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["equipamento", "data_evento"]).reset_index(drop=True)
    return df


def build_historico_recente(maint: pd.DataFrame) -> pd.DataFrame:
    """Última leitura por equipamento (uma linha por equipamento)."""
    if maint.empty:
        return maint
    df = maint.sort_values(["equipamento", "data_ultima_substituicao"], ascending=[True, False])
    return df.drop_duplicates(subset=["equipamento"], keep="first").reset_index(drop=True)


def build_janelas_operacao(historico: pd.DataFrame, producao: pd.DataFrame) -> pd.DataFrame:
    """
    Janelas entre substituições consecutivas, com produção total na janela.
    Filtra apenas eventos de SUBSTITUICAO (não preventivas).
    """
    if historico.empty:
        return pd.DataFrame()
    subs = historico[historico["tipo_evento"].isin(["ULTIMA_SUBSTITUICAO", "PENULTIMA_SUBSTITUICAO"])].copy()
    subs = subs.sort_values(["equipamento", "data_evento"]).reset_index(drop=True)

    rows = []
    for equip, grp in subs.groupby("equipamento"):
        events = grp["data_evento"].tolist()
        for i in range(len(events) - 1):
            inicio, fim = events[i], events[i + 1]
            if pd.isna(inicio) or pd.isna(fim):
                continue
            dias = (fim - inicio).days
            qtd_produzida = 0
            massa_consumida = 0.0
            if not producao.empty:
                mask = (
                    (producao["equipamento"] == equip)
                    & (producao["data_producao"] >= inicio)
                    & (producao["data_producao"] < fim)
                )
                janela = producao[mask]
                if not janela.empty:
                    if "Qtd. Produzida" in janela.columns:
                        qtd_produzida = int(janela["Qtd. Produzida"].fillna(0).sum())
                    if "Consumo de massa no item em (Kg/100pçs)" in janela.columns and "Qtd. Produzida" in janela.columns:
                        # massa estimada em kg = (consumo_kg_por_100pcs * qtd) / 100
                        massa_consumida = float((
                            janela["Consumo de massa no item em (Kg/100pçs)"].fillna(0)
                            * janela["Qtd. Produzida"].fillna(0) / 100
                        ).sum())
            rows.append({
                "equipamento": equip,
                "data_inicio": inicio,
                "data_fim": fim,
                "dias_operacao": dias,
                "qtd_produzida": qtd_produzida,
                "massa_consumida_kg": round(massa_consumida, 2),
            })
    return pd.DataFrame(rows).sort_values(["equipamento", "data_inicio"]).reset_index(drop=True)


def build_ociosidade(historico_recente: pd.DataFrame, producao: pd.DataFrame, hoje: datetime) -> pd.DataFrame:
    """Dias sem produção entre última produção e hoje, por equipamento."""
    rows = []
    if historico_recente.empty:
        return pd.DataFrame()
    for _, r in historico_recente.iterrows():
        equip = r["equipamento"]
        ultima_troca = r.get("data_ultima_substituicao")
        ultima_producao = pd.NaT
        if not producao.empty:
            mask = producao["equipamento"] == equip
            if mask.any():
                ultima_producao = producao.loc[mask, "data_producao"].max()
        if pd.isna(ultima_producao):
            dias_ociosidade = 0
        else:
            dias_ociosidade = max(0, (pd.Timestamp(hoje).normalize() - ultima_producao.normalize()).days)
        rows.append({
            "equipamento": equip,
            "data_ultima_substituicao": ultima_troca,
            "data_ultima_producao": ultima_producao,
            "data_referencia": pd.Timestamp(hoje).normalize(),
            "dias_ociosidade": dias_ociosidade,
        })
    return pd.DataFrame(rows).sort_values("equipamento").reset_index(drop=True)


def main(**kwargs) -> dict:
    print("=" * 60)
    print("ETAPA 7: CRUZAMENTOS HISTÓRICO × PRODUÇÃO")
    print("=" * 60)

    print("\n[1/4] Lendo manutenção e preventivas...")
    maint = _read_maintenance_full()
    prev = _read_preventivas()
    producao = _read_production()
    print(f"  ✓ {len(maint)} equipamentos com manutenção, "
          f"{len(prev)} eventos preventivos, {len(producao)} apontamentos de produção")

    print("\n[2/4] Histórico completo (todos os eventos)...")
    historico = build_historico_completo(maint, prev)
    historico.to_csv(HISTORICO_COMPLETO_FILE, index=False)
    print(f"  ✓ {len(historico)} eventos → {HISTORICO_COMPLETO_FILE.name}")

    print("\n[3/4] Histórico recente (última leitura por equipamento)...")
    recente = build_historico_recente(maint)
    recente.to_csv(HISTORICO_RECENTE_FILE, index=False)
    print(f"  ✓ {len(recente)} equipamentos → {HISTORICO_RECENTE_FILE.name}")

    print("\n[4/4] Janelas de operação e ociosidade...")
    janelas = build_janelas_operacao(historico, producao)
    janelas.to_csv(JANELAS_OPERACAO_FILE, index=False)
    print(f"  ✓ {len(janelas)} janelas → {JANELAS_OPERACAO_FILE.name}")

    hoje = kwargs.get("data_referencia") or datetime.now()
    if isinstance(hoje, str):
        hoje = pd.to_datetime(hoje, dayfirst=True)
    ociosidade = build_ociosidade(recente, producao, hoje)
    ociosidade.to_csv(OCIOSIDADE_FILE, index=False)
    print(f"  ✓ {len(ociosidade)} equipamentos → {OCIOSIDADE_FILE.name}")

    print("\n" + "=" * 60)
    print("ETAPA 7 CONCLUÍDA")
    print("=" * 60)

    return {
        "status": "success",
        "n_eventos": len(historico),
        "n_equipamentos_manut": len(recente),
        "n_janelas": len(janelas),
        "n_equipamentos_ociosidade": len(ociosidade),
    }


if __name__ == "__main__":
    main()
