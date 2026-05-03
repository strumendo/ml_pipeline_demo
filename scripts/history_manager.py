"""
History Manager - Gerenciador de Histórico de Execuções
========================================================
Salva resultados de cada execução de forma segregada com:
- Timestamp único
- Métricas de cada modelo
- Parâmetros utilizados
- Comparativo entre execuções
"""

import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Optional


class HistoryManager:
    """Gerencia o histórico de execuções do pipeline."""

    def __init__(self, history_dir: str = "history"):
        """
        Inicializa o gerenciador de histórico.

        Args:
            history_dir: Diretório para salvar os históricos
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # Criar subdiretórios
        (self.history_dir / "runs").mkdir(exist_ok=True)
        (self.history_dir / "models").mkdir(exist_ok=True)
        (self.history_dir / "reports").mkdir(exist_ok=True)

        # ID da execução atual
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "steps": {},
            "summary": {}
        }

    def log_step(self, step_name: str, results: dict, params: Optional[dict] = None):
        """
        Registra os resultados de um passo do pipeline.

        Args:
            step_name: Nome do passo (ex: "classification", "svr")
            results: Dicionário com resultados dos modelos
            params: Parâmetros utilizados (opcional)
        """
        step_data = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "extras": {},
            "params": params or {},
            "best_model": None,
            "best_mse": float("inf"),
            "best_r2": float("-inf")
        }

        # Processar resultados de cada modelo (e capturar artefatos não-modelo)
        for key, value in results.items():
            if isinstance(value, dict) and "mse" in value:
                model_info = {
                    "mse": value.get("mse"),
                    "r2": value.get("r2"),
                    "accuracy": value.get("accuracy"),
                }
                model_info = {k: v for k, v in model_info.items() if v is not None}
                step_data["models"][key] = model_info

                if value.get("mse") is not None:
                    if value["mse"] < step_data["best_mse"]:
                        step_data["best_mse"] = value["mse"]
                        step_data["best_r2"] = value.get("r2", 0)
                        step_data["best_model"] = key
            elif isinstance(value, (str, int, float, bool, list)) or value is None:
                # Artefato genérico (contagens, paths, status) — útil para estágios não-ML
                step_data["extras"][key] = value

        self.run_data["steps"][step_name] = step_data

        print(f"\n📊 Histórico registrado para: {step_name}")
        if step_data["best_model"]:
            print(f"   Melhor modelo: {step_data['best_model']} (MSE: {step_data['best_mse']:.4f})")
        elif step_data["extras"]:
            n = len(step_data["extras"])
            print(f"   Artefatos registrados: {n} campo(s)")

    def log_classification_step(self, step_name: str, results: dict, params: Optional[dict] = None):
        """
        Registra resultados de um passo de classificação.

        Args:
            step_name: Nome do passo
            results: Dicionário com resultados
            params: Parâmetros utilizados
        """
        step_data = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "params": params or {},
            "best_model": None,
            "best_accuracy": 0
        }

        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                accuracy = model_results.get("accuracy")
                if accuracy is not None:
                    step_data["models"][model_name] = {"accuracy": accuracy}

                    if accuracy > step_data["best_accuracy"]:
                        step_data["best_accuracy"] = accuracy
                        step_data["best_model"] = model_name

        self.run_data["steps"][step_name] = step_data

        print(f"\n📊 Histórico registrado para: {step_name}")
        if step_data["best_model"]:
            print(f"   Melhor modelo: {step_data['best_model']} (Accuracy: {step_data['best_accuracy']:.4f})")

    def save_run(self):
        """Salva os dados da execução atual."""
        # Gerar sumário
        self._generate_summary()

        # Salvar JSON da execução
        run_file = self.history_dir / "runs" / f"run_{self.run_id}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(self.run_data, f, indent=2, ensure_ascii=False)

        # Atualizar índice de execuções
        self._update_index()

        # Gerar relatório
        self._generate_report()

        print(f"\n✅ Execução salva: {run_file}")
        return run_file

    def _generate_summary(self):
        """Gera sumário da execução."""
        summary = {
            "total_steps": len(self.run_data["steps"]),
            "best_models_per_step": {},
            "overall_best": None
        }

        best_overall_mse = float("inf")

        for step_name, step_data in self.run_data["steps"].items():
            if step_data.get("best_model"):
                summary["best_models_per_step"][step_name] = {
                    "model": step_data["best_model"],
                    "mse": step_data.get("best_mse"),
                    "r2": step_data.get("best_r2"),
                    "accuracy": step_data.get("best_accuracy")
                }

                if step_data.get("best_mse") and step_data["best_mse"] < best_overall_mse:
                    best_overall_mse = step_data["best_mse"]
                    summary["overall_best"] = {
                        "step": step_name,
                        "model": step_data["best_model"],
                        "mse": step_data["best_mse"]
                    }

        self.run_data["summary"] = summary

    def _update_index(self):
        """Atualiza o índice de todas as execuções."""
        index_file = self.history_dir / "index.json"

        # Carregar índice existente
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
        else:
            index = {"runs": []}

        # Adicionar nova execução
        index["runs"].append({
            "run_id": self.run_id,
            "timestamp": self.run_data["timestamp"],
            "total_steps": self.run_data["summary"]["total_steps"],
            "overall_best": self.run_data["summary"].get("overall_best")
        })

        # Salvar índice atualizado
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    def _generate_report(self):
        """Gera relatório em texto da execução."""
        report_file = self.history_dir / "reports" / f"report_{self.run_id}.txt"

        lines = [
            "=" * 70,
            f"RELATÓRIO DE EXECUÇÃO - {self.run_id}",
            "=" * 70,
            f"Data/Hora: {self.run_data['timestamp']}",
            f"Total de passos: {self.run_data['summary']['total_steps']}",
            "",
            "-" * 70,
            "RESULTADOS POR PASSO",
            "-" * 70,
        ]

        for step_name, step_data in self.run_data["steps"].items():
            lines.append(f"\n📌 {step_name.upper()}")
            lines.append(f"   Timestamp: {step_data['timestamp']}")

            if step_data.get("models"):
                lines.append("   Modelos avaliados:")
                for model_name, metrics in step_data["models"].items():
                    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    lines.append(f"     - {model_name}: {metrics_str}")

            if step_data.get("best_model"):
                lines.append(f"   ⭐ Melhor: {step_data['best_model']}")

        # Sumário
        lines.extend([
            "",
            "-" * 70,
            "SUMÁRIO",
            "-" * 70,
        ])

        if self.run_data["summary"].get("overall_best"):
            best = self.run_data["summary"]["overall_best"]
            lines.append(f"Melhor modelo geral: {best['model']} ({best['step']})")
            lines.append(f"MSE: {best['mse']:.4f}")

        lines.append("\n" + "=" * 70)

        # Salvar relatório
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"📄 Relatório salvo: {report_file}")

    def compare_runs(self, n_runs: int = 5) -> pd.DataFrame:
        """
        Compara as últimas N execuções.

        Args:
            n_runs: Número de execuções a comparar

        Returns:
            DataFrame com comparativo
        """
        index_file = self.history_dir / "index.json"

        if not index_file.exists():
            print("Nenhuma execução anterior encontrada.")
            return pd.DataFrame()

        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)

        # Pegar últimas N execuções
        recent_runs = index["runs"][-n_runs:]

        # Montar dados para comparação
        comparison_data = []
        for run_info in recent_runs:
            run_file = self.history_dir / "runs" / f"run_{run_info['run_id']}.json"
            if run_file.exists():
                with open(run_file, "r", encoding="utf-8") as f:
                    run_data = json.load(f)

                row = {
                    "run_id": run_info["run_id"],
                    "timestamp": run_info["timestamp"][:19],
                }

                # Adicionar métricas de cada passo
                for step_name, step_data in run_data.get("steps", {}).items():
                    if step_data.get("best_mse"):
                        row[f"{step_name}_mse"] = step_data["best_mse"]
                        row[f"{step_name}_model"] = step_data.get("best_model", "")

                comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df

    def get_best_run(self) -> Optional[dict]:
        """Retorna a execução com melhor resultado geral."""
        index_file = self.history_dir / "index.json"

        if not index_file.exists():
            return None

        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)

        best_run = None
        best_mse = float("inf")

        for run_info in index["runs"]:
            if run_info.get("overall_best") and run_info["overall_best"].get("mse"):
                if run_info["overall_best"]["mse"] < best_mse:
                    best_mse = run_info["overall_best"]["mse"]
                    best_run = run_info

        return best_run


def print_history_summary(history_dir: str = "history"):
    """Imprime resumo do histórico de execuções."""
    hm = HistoryManager(history_dir)
    df = hm.compare_runs(n_runs=10)

    if df.empty:
        print("Nenhuma execução encontrada no histórico.")
        return

    print("\n" + "=" * 70)
    print("HISTÓRICO DE EXECUÇÕES")
    print("=" * 70)
    print(df.to_markdown(index=False))

    best = hm.get_best_run()
    if best:
        print(f"\n⭐ Melhor execução: {best['run_id']}")
        print(f"   Modelo: {best['overall_best']['model']}")
        print(f"   MSE: {best['overall_best']['mse']:.4f}")


if __name__ == "__main__":
    # Teste do gerenciador
    print_history_summary()
