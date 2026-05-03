"""
S08 - Prescrição da Próxima Manutenção
=======================================
Etapa 8 do Pipeline (opcional)

O QUE FAZ:
Aplica a fórmula prescritiva, combinando 3 sinais:

    T_base          = mediana(dias_em_operacao) por equipamento, ou 450 dias (fallback)
    fator_desgaste  = clamp( 1 / (amp_atual / amp_mediana) , 0.60 , 1.20 )
                      onde amp = max-min do componente; média de A e B
    fator_massa     = clamp( 1 / (massa_atual / massa_mediana_janelas) , 0.70 , 1.30 )
    T_prescrito     = T_base × fator_desgaste × fator_massa
    data_prescrita  = data_ultima_substituicao + T_prescrito + dias_ociosidade

Buckets de urgência (sobre dias_restantes = data_prescrita - hoje):
- ATRASADO  (< 0)
- URGENTE   (0–29)
- ATENÇÃO   (30–89)
- OK        (≥ 90)

ENTRADAS:
- outputs/historico_recente.csv   (s07)
- outputs/janelas_operacao.csv    (s07)
- outputs/ociosidade.csv          (s07)
- data/raw/EQ-*.csv               (apontamentos para massa atual)

SAÍDA:
- outputs/prescricao_manutencao.csv
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
        DATA_RAW_DIR,
        HISTORICO_RECENTE_FILE, JANELAS_OPERACAO_FILE, OCIOSIDADE_FILE,
        PRESCRICAO_MANUTENCAO_FILE,
    )
except ImportError:
    DATA_RAW_DIR = BASE_DIR / "data" / "raw"
    HISTORICO_RECENTE_FILE = Path("historico_recente.csv")
    JANELAS_OPERACAO_FILE = Path("janelas_operacao.csv")
    OCIOSIDADE_FILE = Path("ociosidade.csv")
    PRESCRICAO_MANUTENCAO_FILE = Path("prescricao_manutencao.csv")

T_BASE_FALLBACK = 450
CLAMP_DESGASTE = (0.60, 1.20)
CLAMP_MASSA = (0.70, 1.30)


def _clamp(x, lo, hi):
    if pd.isna(x):
        return None
    return max(lo, min(hi, x))


def _classify(dias_restantes: float) -> str:
    if pd.isna(dias_restantes):
        return "DESCONHECIDO"
    if dias_restantes < 0:
        return "ATRASADO"
    if dias_restantes < 30:
        return "URGENTE"
    if dias_restantes < 90:
        return "ATENÇÃO"
    return "OK"


def _massa_pos_ultima_troca(producao: pd.DataFrame, equip: str, ultima_troca) -> float:
    if producao.empty or pd.isna(ultima_troca):
        return 0.0
    mask = (producao["equipamento"] == equip) & (producao["data_producao"] >= ultima_troca)
    janela = producao[mask]
    if janela.empty:
        return 0.0
    return float((
        janela["Consumo de massa no item em (Kg/100pçs)"].fillna(0)
        * janela["Qtd. Produzida"].fillna(0) / 100
    ).sum())


def _read_production() -> pd.DataFrame:
    files = sorted(DATA_RAW_DIR.glob("EQ-*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["data_producao"] = pd.to_datetime(df["Data de Produção"], dayfirst=True, errors="coerce")
    df["equipamento"] = df["Cód. Recurso"].astype(str).str.strip()
    return df


def main(**kwargs) -> dict:
    print("=" * 60)
    print("ETAPA 8: PRESCRIÇÃO DA PRÓXIMA MANUTENÇÃO")
    print("=" * 60)

    if not HISTORICO_RECENTE_FILE.exists():
        print(f"\n✗ Arquivo não encontrado: {HISTORICO_RECENTE_FILE}")
        print("Execute a Etapa 7 primeiro (s07_cross_reference.py)")
        return {"status": "error", "message": "historico_recente.csv missing"}

    recente = pd.read_csv(HISTORICO_RECENTE_FILE, parse_dates=[
        "data_ultima_substituicao", "data_penultima_substituicao",
    ])
    janelas = pd.read_csv(JANELAS_OPERACAO_FILE, parse_dates=["data_inicio", "data_fim"]) \
        if JANELAS_OPERACAO_FILE.exists() else pd.DataFrame()
    ociosidade = pd.read_csv(OCIOSIDADE_FILE, parse_dates=[
        "data_ultima_substituicao", "data_ultima_producao", "data_referencia",
    ]) if OCIOSIDADE_FILE.exists() else pd.DataFrame()
    producao = _read_production()

    hoje = kwargs.get("data_referencia") or datetime.now()
    if isinstance(hoje, str):
        hoje = pd.to_datetime(hoje, dayfirst=True)
    hoje = pd.Timestamp(hoje).normalize()

    rows = []
    for _, r in recente.iterrows():
        equip = r["equipamento"]
        ultima_troca = r["data_ultima_substituicao"]

        # T_base — mediana das janelas do equipamento; fallback 450
        if not janelas.empty:
            grp = janelas[janelas["equipamento"] == equip]
            t_base = float(grp["dias_operacao"].median()) if not grp.empty else T_BASE_FALLBACK
        else:
            t_base = T_BASE_FALLBACK
        if pd.isna(t_base):
            t_base = T_BASE_FALLBACK

        # Amplitudes atuais
        amp_a_atual = (r.get("componente_a_max") or 0) - (r.get("componente_a_min") or 0) \
            if pd.notna(r.get("componente_a_max")) and pd.notna(r.get("componente_a_min")) else None
        amp_b_atual = (r.get("componente_b_max") or 0) - (r.get("componente_b_min") or 0) \
            if pd.notna(r.get("componente_b_max")) and pd.notna(r.get("componente_b_min")) else None

        # Mediana de amplitudes históricas — usa o próprio historico_recente como aproximação
        amp_a_med = float((recente["componente_a_max"] - recente["componente_a_min"]).median())
        amp_b_med = float((recente["componente_b_max"] - recente["componente_b_min"]).median())

        fator_desg_partes = []
        if amp_a_atual is not None and amp_a_med and not np.isclose(amp_a_med, 0):
            fator_desg_partes.append(amp_a_med / amp_a_atual if amp_a_atual else 1.0)
        if amp_b_atual is not None and amp_b_med and not np.isclose(amp_b_med, 0):
            fator_desg_partes.append(amp_b_med / amp_b_atual if amp_b_atual else 1.0)
        fator_desgaste = float(np.mean(fator_desg_partes)) if fator_desg_partes else 1.0
        fator_desgaste = _clamp(fator_desgaste, *CLAMP_DESGASTE) or 1.0

        # Massa atual (pós última troca) vs mediana das janelas
        massa_atual = _massa_pos_ultima_troca(producao, equip, ultima_troca)
        massa_med_janelas = 0.0
        if not janelas.empty:
            grp = janelas[janelas["equipamento"] == equip]
            if not grp.empty:
                massa_med_janelas = float(grp["massa_consumida_kg"].median())
        if massa_med_janelas > 0 and massa_atual > 0:
            fator_massa = _clamp(massa_med_janelas / massa_atual, *CLAMP_MASSA) or 1.0
        else:
            fator_massa = 1.0

        t_prescrito = t_base * fator_desgaste * fator_massa

        dias_ociosidade = 0
        if not ociosidade.empty:
            grp = ociosidade[ociosidade["equipamento"] == equip]
            if not grp.empty:
                dias_ociosidade = int(grp["dias_ociosidade"].iloc[0])

        if pd.notna(ultima_troca):
            data_prescrita = pd.Timestamp(ultima_troca).normalize() \
                + pd.Timedelta(days=int(round(t_prescrito))) \
                + pd.Timedelta(days=dias_ociosidade)
            dias_restantes = (data_prescrita - hoje).days
        else:
            data_prescrita = pd.NaT
            dias_restantes = None

        rows.append({
            "equipamento": equip,
            "data_ultima_substituicao": ultima_troca,
            "T_base_dias": round(t_base, 1),
            "amplitude_a_atual": round(amp_a_atual, 4) if amp_a_atual is not None else None,
            "amplitude_b_atual": round(amp_b_atual, 4) if amp_b_atual is not None else None,
            "fator_desgaste": round(fator_desgaste, 4),
            "massa_atual_kg": round(massa_atual, 2),
            "massa_mediana_janelas_kg": round(massa_med_janelas, 2),
            "fator_massa": round(fator_massa, 4),
            "T_prescrito_dias": round(t_prescrito, 1),
            "dias_ociosidade": dias_ociosidade,
            "data_prescrita": data_prescrita,
            "data_referencia": hoje,
            "dias_restantes": dias_restantes,
            "urgencia": _classify(dias_restantes if dias_restantes is not None else float("nan")),
        })

    df = pd.DataFrame(rows).sort_values("dias_restantes", na_position="last").reset_index(drop=True)
    df.to_csv(PRESCRICAO_MANUTENCAO_FILE, index=False)

    print(f"\n  ✓ Prescrição gerada para {len(df)} equipamentos → {PRESCRICAO_MANUTENCAO_FILE.name}")
    counts = df["urgencia"].value_counts().to_dict()
    print(f"  Distribuição: {counts}")

    print("\n" + "=" * 60)
    print("ETAPA 8 CONCLUÍDA")
    print("=" * 60)

    return {
        "status": "success",
        "n_equipamentos": len(df),
        "distribuicao_urgencia": counts,
        "output_file": str(PRESCRICAO_MANUTENCAO_FILE),
    }


if __name__ == "__main__":
    main()
