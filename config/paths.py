"""
Configuração de caminhos do Pipeline DemoML
===================================================
Centraliza todos os caminhos utilizados pelos scripts.
"""

from pathlib import Path

# Diretório base
BASE_DIR = Path(__file__).parent.parent

# Diretórios principais
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_MANUTENCAO_DIR = DATA_DIR / "manutencao"  # Nova pasta de manutenção
DATA_ARQUIVO_UNICO_DIR = DATA_DIR / "arquivo_unico"
DATA_ARQUIVO_UNICO_PROCESSADO_DIR = DATA_DIR / "arquivo_unico_processado"
SCRIPTS_DIR = BASE_DIR / "scripts"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIG_DIR = BASE_DIR / "config"

# Subdiretórios de outputs
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"
HISTORY_DIR = OUTPUTS_DIR / "history"


# Arquivos intermediários (gerados durante o pipeline)
DATA_RAW_FILE = OUTPUTS_DIR / "data_raw.csv"
DATA_PREPROCESSED_FILE = OUTPUTS_DIR / "data_preprocessed.csv"
DATA_EDA_FILE = OUTPUTS_DIR / "data_eda.csv"
TRAIN_TEST_FILE = OUTPUTS_DIR / "train_test_split.npz"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
EDA_REPORT_FILE = OUTPUTS_DIR / "eda_report.txt"
EVALUATION_REPORT_FILE = OUTPUTS_DIR / "evaluation_report.txt"

# Arquivo de estado para automação
DATA_STATE_FILE = OUTPUTS_DIR / ".data_state.json"


def ensure_directories():
    """Cria todos os diretórios necessários se não existirem."""
    dirs = [
        DATA_DIR, DATA_RAW_DIR, OUTPUTS_DIR, MODELS_DIR,
        PLOTS_DIR, REPORTS_DIR, HISTORY_DIR, CONFIG_DIR,
        DATA_ARQUIVO_UNICO_DIR, DATA_ARQUIVO_UNICO_PROCESSADO_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_raw_data_files(pattern: str = "EQ-*.csv") -> list:
    """Retorna lista de arquivos de dados brutos."""
    return sorted(DATA_RAW_DIR.glob(pattern))


def get_maintenance_file() -> Path:
    """
    Retorna o caminho do arquivo de manutenção mais recente.

    Procura arquivos no padrão "Dados Manut*.xlsx" na pasta data/manutencao/
    ou data/ (fallback).

    Returns:
        Path do arquivo ou None se não encontrado
    """
    # Primeiro, procurar na nova pasta data/manutencao/
    if DATA_MANUTENCAO_DIR.exists():
        maint_files = list(DATA_MANUTENCAO_DIR.glob("Dados Manut*.xlsx"))
        if maint_files:
            return max(maint_files, key=lambda f: f.stat().st_mtime)

    # Fallback para pasta data/ (compatibilidade)
    maint_files = list(DATA_DIR.glob("Dados Manut*.xlsx"))
    if maint_files:
        return max(maint_files, key=lambda f: f.stat().st_mtime)

    return None


def get_maintenance_history_file() -> Path:
    """
    Retorna o caminho do arquivo CSV de histórico de manutenção.

    Procura arquivos CSV na pasta data/manutencao/.

    Returns:
        Path do arquivo ou None se não encontrado
    """
    if DATA_MANUTENCAO_DIR.exists():
        csv_files = list(DATA_MANUTENCAO_DIR.glob("*.csv"))
        if csv_files:
            return max(csv_files, key=lambda f: f.stat().st_mtime)
    return None


def get_all_maintenance_files() -> dict:
    """
    Retorna todos os arquivos de manutenção disponíveis.

    Returns:
        Dict com categorias: xlsx, csv
    """
    files = {"xlsx": [], "csv": []}

    if DATA_MANUTENCAO_DIR.exists():
        files["xlsx"] = sorted(DATA_MANUTENCAO_DIR.glob("*.xlsx"))
        files["csv"] = sorted(DATA_MANUTENCAO_DIR.glob("*.csv"))

    # Incluir arquivos da pasta data/ (compatibilidade)
    files["xlsx"].extend(list(DATA_DIR.glob("Dados Manut*.xlsx")))

    return files


def get_all_data_files() -> dict:
    """
    Retorna dicionário com todos os arquivos de dados.

    Returns:
        Dict com categorias: raw_xlsx, raw_csv, producao, manutencao_xlsx, manutencao_csv
    """
    maint_files = get_all_maintenance_files()

    files = {
        "raw_xlsx": sorted(DATA_RAW_DIR.glob("EQ-*.xlsx")),
        "raw_csv": sorted(DATA_RAW_DIR.glob("EQ-*.csv")),
        "producao": sorted(DATA_RAW_DIR.glob("DadosProducao*.xlsx")),
        "manutencao_xlsx": maint_files.get("xlsx", []),
        "manutencao_csv": maint_files.get("csv", []),
    }
    return files


# Criar diretórios ao importar
ensure_directories()
