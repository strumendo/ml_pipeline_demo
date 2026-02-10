"""
DemoML - Pipeline Automático com Detecção de Alterações
======================================================
Monitora a pasta de dados e executa automaticamente o pipeline
quando novos arquivos são adicionados ou arquivos existentes são atualizados.

Uso:
    python auto_pipeline.py                    # Executa uma vez (detecta e processa)
    python auto_pipeline.py --watch            # Modo contínuo (monitora alterações)
    python auto_pipeline.py --watch --interval 60  # Verifica a cada 60 segundos
    python auto_pipeline.py --status           # Mostra status dos arquivos
    python auto_pipeline.py --force            # Força re-execução do pipeline
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Diretório base
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

# Adicionar diretórios ao path
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BASE_DIR / "config"))

try:
    from paths import (
        DATA_RAW_DIR, DATA_DIR, DATA_MANUTENCAO_DIR, OUTPUTS_DIR,
        DATA_ARQUIVO_UNICO_DIR
    )
except ImportError:
    DATA_DIR = BASE_DIR / "data"
    DATA_RAW_DIR = DATA_DIR / "raw"
    DATA_MANUTENCAO_DIR = DATA_DIR / "manutencao"
    DATA_ARQUIVO_UNICO_DIR = DATA_DIR / "arquivo_unico"
    OUTPUTS_DIR = BASE_DIR / "outputs"

# Arquivo de estado (guarda hashes dos arquivos)
STATE_FILE = OUTPUTS_DIR / ".data_state.json"


def calculate_file_hash(filepath: Path) -> str:
    """
    Calcula hash MD5 de um arquivo.

    Args:
        filepath: Caminho do arquivo

    Returns:
        Hash MD5 em hexadecimal
    """
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"  Erro ao calcular hash de {filepath.name}: {e}")
        return ""


def get_file_info(filepath: Path) -> dict:
    """
    Obtém informações de um arquivo (tamanho, modificação, hash).

    Args:
        filepath: Caminho do arquivo

    Returns:
        Dicionário com informações do arquivo
    """
    stat = filepath.stat()
    return {
        "name": filepath.name,
        "path": str(filepath),
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "hash": calculate_file_hash(filepath)
    }


def scan_data_files() -> dict:
    """
    Escaneia todos os arquivos de dados (XLSX e CSV).

    Returns:
        Dicionário com informações de todos os arquivos
    """
    files_info = {
        "raw_files": {},
        "maintenance_files": {},
        "arquivo_unico_files": {},
        "scan_time": datetime.now().isoformat()
    }

    # Escanear arquivos na pasta raw (EQ-*.xlsx, DadosProducao*.xlsx)
    if DATA_RAW_DIR.exists():
        patterns = ["EQ-*.xlsx", "EQ-*.csv", "DadosProducao*.xlsx"]
        for pattern in patterns:
            for filepath in DATA_RAW_DIR.glob(pattern):
                files_info["raw_files"][filepath.name] = get_file_info(filepath)

    # Escanear arquivos de manutenção na pasta data/manutencao/ (NOVA PASTA)
    if DATA_MANUTENCAO_DIR.exists():
        for pattern in ["*.xlsx", "*.csv"]:
            for filepath in DATA_MANUTENCAO_DIR.glob(pattern):
                files_info["maintenance_files"][filepath.name] = get_file_info(filepath)

    # Fallback: Escanear arquivos de manutenção na pasta data/ (compatibilidade)
    if DATA_DIR.exists():
        for pattern in ["Dados Manut*.xlsx"]:
            for filepath in DATA_DIR.glob(pattern):
                # Ignorar arquivos em subpastas
                if filepath.parent == DATA_DIR:
                    files_info["maintenance_files"][filepath.name] = get_file_info(filepath)

    # Escanear arquivos unificados na pasta data/arquivo_unico/
    if DATA_ARQUIVO_UNICO_DIR.exists():
        for filepath in DATA_ARQUIVO_UNICO_DIR.glob("*.xlsx"):
            files_info["arquivo_unico_files"][filepath.name] = get_file_info(filepath)

    return files_info


def load_state() -> dict:
    """
    Carrega o estado anterior dos arquivos.

    Returns:
        Dicionário com estado anterior ou vazio se não existir
    """
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Aviso: Erro ao carregar estado anterior: {e}")
    return {"raw_files": {}, "maintenance_files": {}, "arquivo_unico_files": {}}


def save_state(state: dict):
    """
    Salva o estado atual dos arquivos.

    Args:
        state: Dicionário com estado dos arquivos
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Erro ao salvar estado: {e}")


def detect_changes(old_state: dict, new_state: dict) -> dict:
    """
    Detecta alterações entre dois estados.

    Args:
        old_state: Estado anterior
        new_state: Estado atual

    Returns:
        Dicionário com arquivos novos, modificados e removidos
    """
    changes = {
        "new_files": [],
        "modified_files": [],
        "removed_files": [],
        "has_changes": False
    }

    # Combinar raw_files, maintenance_files e arquivo_unico_files
    for category in ["raw_files", "maintenance_files", "arquivo_unico_files"]:
        old_files = old_state.get(category, {})
        new_files = new_state.get(category, {})

        # Arquivos novos
        for name, info in new_files.items():
            if name not in old_files:
                changes["new_files"].append({"name": name, "category": category, **info})

        # Arquivos modificados (hash diferente)
        for name, info in new_files.items():
            if name in old_files:
                if info["hash"] != old_files[name]["hash"]:
                    changes["modified_files"].append({
                        "name": name,
                        "category": category,
                        "old_modified": old_files[name].get("modified_date"),
                        "new_modified": info.get("modified_date"),
                        **info
                    })

        # Arquivos removidos
        for name in old_files:
            if name not in new_files:
                changes["removed_files"].append({"name": name, "category": category})

    changes["has_changes"] = bool(
        changes["new_files"] or
        changes["modified_files"] or
        changes["removed_files"]
    )

    return changes


def print_changes(changes: dict):
    """
    Imprime as alterações detectadas de forma formatada.

    Args:
        changes: Dicionário com alterações
    """
    print("\n" + "=" * 60)
    print("ALTERAÇÕES DETECTADAS")
    print("=" * 60)

    if not changes["has_changes"]:
        print("\n✓ Nenhuma alteração detectada nos arquivos de dados.")
        return

    if changes["new_files"]:
        print(f"\n📥 ARQUIVOS NOVOS ({len(changes['new_files'])}):")
        for f in changes["new_files"]:
            size_kb = f["size"] / 1024
            print(f"   + {f['name']} ({size_kb:.1f} KB)")

    if changes["modified_files"]:
        print(f"\n📝 ARQUIVOS MODIFICADOS ({len(changes['modified_files'])}):")
        for f in changes["modified_files"]:
            print(f"   ~ {f['name']}")
            print(f"     Anterior: {f['old_modified']}")
            print(f"     Atual:    {f['new_modified']}")

    if changes["removed_files"]:
        print(f"\n🗑️  ARQUIVOS REMOVIDOS ({len(changes['removed_files'])}):")
        for f in changes["removed_files"]:
            print(f"   - {f['name']}")

    print()


def print_status(current_state: dict, old_state: dict = None):
    """
    Imprime status atual dos arquivos monitorados.

    Args:
        current_state: Estado atual
        old_state: Estado anterior (opcional)
    """
    print("\n" + "=" * 70)
    print("STATUS DOS ARQUIVOS DE DADOS")
    print("=" * 70)

    print(f"\nÚltimo scan: {current_state.get('scan_time', 'N/A')}")

    # Arquivos raw
    raw_files = current_state.get("raw_files", {})
    print(f"\n📁 PASTA DATA/RAW ({len(raw_files)} arquivos):")
    if raw_files:
        for name, info in sorted(raw_files.items()):
            size_kb = info["size"] / 1024
            mod_date = info.get("modified_date", "")[:19]  # Truncar microsegundos
            print(f"   {name:<40} {size_kb:>8.1f} KB  {mod_date}")
    else:
        print("   (vazio)")

    # Arquivos de manutenção
    maint_files = current_state.get("maintenance_files", {})
    print(f"\n📁 PASTA DATA (manutenção) ({len(maint_files)} arquivos):")
    if maint_files:
        for name, info in sorted(maint_files.items()):
            size_kb = info["size"] / 1024
            mod_date = info.get("modified_date", "")[:19]
            print(f"   {name:<40} {size_kb:>8.1f} KB  {mod_date}")
    else:
        print("   (vazio)")

    # Comparar com estado anterior
    if old_state:
        changes = detect_changes(old_state, current_state)
        if changes["has_changes"]:
            print_changes(changes)
        else:
            print("\n✓ Nenhuma alteração desde a última execução.")

    print()


def run_pipeline():
    """
    Executa o pipeline completo.

    Returns:
        True se executou com sucesso, False caso contrário
    """
    print("\n" + "=" * 60)
    print("EXECUTANDO PIPELINE AUTOMÁTICO")
    print("=" * 60)

    try:
        # Mudar para o diretório de outputs
        original_dir = os.getcwd()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        os.chdir(str(OUTPUTS_DIR))

        # Importar e executar o pipeline
        from run_pipeline import run_full_pipeline
        results = run_full_pipeline(save_history=True)

        # Voltar ao diretório original
        os.chdir(original_dir)

        # Verificar se teve sucesso
        successful = sum(1 for r in results.values() if r and r.get("status") != "error")
        return successful >= 5

    except Exception as e:
        print(f"\n✗ Erro ao executar pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def auto_run(force: bool = False) -> bool:
    """
    Verifica alterações e executa pipeline se necessário.

    Args:
        force: Se True, força execução mesmo sem alterações

    Returns:
        True se pipeline foi executado
    """
    print("\n" + "=" * 60)
    print("DemoML - VERIFICAÇÃO AUTOMÁTICA DE ALTERAÇÕES")
    print(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 60)

    # Carregar estado anterior
    old_state = load_state()

    # Escanear arquivos atuais
    print("\nEscaneando arquivos de dados...")
    current_state = scan_data_files()

    # Contar arquivos
    total_raw = len(current_state.get("raw_files", {}))
    total_maint = len(current_state.get("maintenance_files", {}))
    print(f"  Arquivos em data/raw: {total_raw}")
    print(f"  Arquivos de manutenção: {total_maint}")

    # Detectar alterações
    changes = detect_changes(old_state, current_state)

    if force:
        print("\n⚠ Modo forçado: Pipeline será executado independente de alterações.")
        run_pipeline()
        save_state(current_state)
        return True

    if changes["has_changes"]:
        print_changes(changes)

        # Executar pipeline
        print("\n🔄 Alterações detectadas! Iniciando pipeline...")
        success = run_pipeline()

        if success:
            # Salvar novo estado
            save_state(current_state)
            print("\n✓ Pipeline executado e estado atualizado.")
        else:
            print("\n✗ Pipeline falhou. Estado não foi atualizado.")

        return True
    else:
        print("\n✓ Nenhuma alteração detectada. Pipeline não será executado.")
        return False


def watch_mode(interval: int = 300):
    """
    Modo de monitoramento contínuo.

    Args:
        interval: Intervalo entre verificações em segundos (padrão: 5 min)
    """
    print("\n" + "=" * 60)
    print("DemoML - MODO DE MONITORAMENTO CONTÍNUO")
    print("=" * 60)
    print(f"\nIntervalo de verificação: {interval} segundos")
    print("Pressione Ctrl+C para sair.\n")

    try:
        while True:
            auto_run()

            print(f"\nPróxima verificação em {interval} segundos...")
            print("-" * 40)
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n✓ Monitoramento encerrado pelo usuário.")


def main():
    parser = argparse.ArgumentParser(
        description="DemoML - Pipeline Automático com Detecção de Alterações",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python auto_pipeline.py                     # Verifica alterações e executa se necessário
  python auto_pipeline.py --status            # Mostra status dos arquivos
  python auto_pipeline.py --force             # Força execução do pipeline
  python auto_pipeline.py --watch             # Modo monitoramento contínuo (5 min)
  python auto_pipeline.py --watch --interval 60  # Verifica a cada 60 segundos

O sistema detecta automaticamente:
  - Novos arquivos XLSX/CSV na pasta data/raw/
  - Arquivos existentes que foram modificados
  - Arquivo de manutenção (Dados Manut*.xlsx) na pasta data/
  - Arquivos DadosProducao*.xlsx com dados consolidados

Quando alterações são detectadas, o pipeline completo é executado:
  1. Coleta e Integração
  2. Pré-processamento
  3. Análise Exploratória (EDA)
  4. Modelagem
  5. Avaliação
  6. Geração de Relatório PDF
        """
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Mostra status atual dos arquivos monitorados"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força execução do pipeline mesmo sem alterações"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Modo de monitoramento contínuo"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Intervalo entre verificações em segundos (padrão: 300)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Limpa o estado salvo (próxima execução reprocessará tudo)"
    )

    args = parser.parse_args()

    # Reset do estado
    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            print("✓ Estado resetado. Próxima execução reprocessará todos os arquivos.")
        else:
            print("Nenhum estado para resetar.")
        return

    # Mostrar status
    if args.status:
        current_state = scan_data_files()
        old_state = load_state()
        print_status(current_state, old_state)
        return

    # Modo watch
    if args.watch:
        watch_mode(args.interval)
        return

    # Execução única
    auto_run(force=args.force)


if __name__ == "__main__":
    main()
