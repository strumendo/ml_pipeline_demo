"""
S05 - Validação e Avaliação
===========================
Etapa 5 do Pipeline conforme fluxos.drawio

O QUE FAZ:
- Testar nos 20% de Dados (conjunto de teste)
- Calcular Métricas: R², MSE, MAE
- Comparar Desempenho entre modelos
- Selecionar Melhor Modelo ou Ajustar Hiperparâmetros

FLUXO (fluxos.drawio):
Modelos Treinados → Testar 20% → Métricas (R², MSE, MAE) → Comparar → Selecionar Melhor

ENTRADA:
- models/*.joblib (modelos da Etapa 4)
- train_test_split.npz (dados de teste da Etapa 4)

SAÍDA:
- best_model.joblib: Melhor modelo selecionado
- evaluation_report.txt: Relatório de avaliação
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_test_data(filepath: str = "train_test_split.npz") -> tuple:
    """
    Carrega dados de teste da Etapa 4.

    Args:
        filepath: Caminho do arquivo .npz

    Returns:
        Tupla (X_test, y_test, feature_names)
    """
    data = np.load(filepath, allow_pickle=True)

    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()

    print(f"  Carregado: {len(X_test)} amostras de teste, {len(feature_names)} features")

    return X_test, y_test, feature_names


def load_models(models_dir: str = "models") -> dict:
    """
    Carrega todos os modelos treinados.

    Args:
        models_dir: Diretório com os modelos

    Returns:
        Dicionário {nome: modelo}
    """
    models = {}
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"  ✗ Diretório não encontrado: {models_dir}")
        return models

    for filepath in models_path.glob("model_*.joblib"):
        model_name = filepath.stem.replace("model_", "")
        models[model_name] = joblib.load(filepath)
        print(f"  ✓ Carregado: {model_name}")

    return models


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Avalia um modelo calculando R², MSE, MAE.

    Métricas conforme fluxos.drawio.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste

    Returns:
        Dicionário com métricas
    """
    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }

    return metrics


def compare_models(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Compara desempenho entre todos os modelos.

    Args:
        models: Dicionário com modelos
        X_test: Features de teste
        y_test: Target de teste

    Returns:
        DataFrame com comparativo
    """
    results = []

    for name, model in models.items():
        print(f"\n  Avaliando {name}...")
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)

        print(f"    R²:   {metrics['r2']:.4f}")
        print(f"    MSE:  {metrics['mse']:.2f}")
        print(f"    MAE:  {metrics['mae']:.2f}")
        print(f"    RMSE: {metrics['rmse']:.2f}")

    df_results = pd.DataFrame(results)
    df_results = df_results[["model", "r2", "mse", "mae", "rmse"]]

    return df_results


def select_best_model(df_results: pd.DataFrame, models: dict, criterion: str = "r2") -> tuple:
    """
    Seleciona o melhor modelo baseado no critério.

    Args:
        df_results: DataFrame com resultados
        models: Dicionário com modelos
        criterion: Critério de seleção ("r2", "mse", "mae")

    Returns:
        Tupla (nome_modelo, modelo, métricas)
    """
    if criterion == "r2":
        # Maior R² é melhor
        best_idx = df_results["r2"].idxmax()
    else:
        # Menor MSE/MAE é melhor
        best_idx = df_results[criterion].idxmin()

    best_row = df_results.iloc[best_idx]
    best_name = best_row["model"]
    best_model = models[best_name]

    best_metrics = {
        "r2": best_row["r2"],
        "mse": best_row["mse"],
        "mae": best_row["mae"],
        "rmse": best_row["rmse"],
    }

    return best_name, best_model, best_metrics


def save_best_model(model, model_name: str, metrics: dict, output_path: str = "best_model.joblib"):
    """
    Salva o melhor modelo.

    Args:
        model: Modelo a salvar
        model_name: Nome do modelo
        metrics: Métricas do modelo
        output_path: Caminho de saída
    """
    # Salvar modelo com metadados
    model_data = {
        "model": model,
        "name": model_name,
        "metrics": metrics,
    }

    joblib.dump(model_data, output_path)
    print(f"  ✓ Melhor modelo salvo: {output_path}")


def generate_evaluation_report(df_results: pd.DataFrame, best_name: str, best_metrics: dict, output_path: str = "evaluation_report.txt"):
    """
    Gera relatório de avaliação.

    Args:
        df_results: DataFrame com resultados
        best_name: Nome do melhor modelo
        best_metrics: Métricas do melhor modelo
        output_path: Caminho do arquivo
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE VALIDAÇÃO E AVALIAÇÃO\n")
        f.write("(Conforme fluxos.drawio - Etapa 5)\n")
        f.write("=" * 70 + "\n\n")

        f.write("MÉTRICAS: R², MSE, MAE\n")
        f.write("-" * 40 + "\n\n")

        f.write("COMPARATIVO DE DESEMPENHO\n")
        f.write("-" * 40 + "\n")

        # Ordenar por R² decrescente
        df_sorted = df_results.sort_values("r2", ascending=False)

        f.write(f"\n{'Rank':<6} {'Modelo':<20} {'R²':<10} {'MSE':<12} {'MAE':<12} {'RMSE':<12}\n")
        f.write("-" * 70 + "\n")

        for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
            marker = "★" if row["model"] == best_name else " "
            f.write(f"{rank:<6} {row['model']:<20} {row['r2']:.4f}     {row['mse']:<12.2f} {row['mae']:<12.2f} {row['rmse']:<12.2f} {marker}\n")

        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write(f"MELHOR MODELO SELECIONADO: {best_name.upper()}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Métricas Finais:\n")
        f.write(f"  R² (Coeficiente de Determinação): {best_metrics['r2']:.4f}\n")
        f.write(f"  MSE (Erro Quadrático Médio):      {best_metrics['mse']:.2f}\n")
        f.write(f"  MAE (Erro Absoluto Médio):        {best_metrics['mae']:.2f}\n")
        f.write(f"  RMSE (Raiz do MSE):               {best_metrics['rmse']:.2f}\n")

        f.write("\n\nInterpretação:\n")
        if best_metrics['r2'] >= 0.9:
            f.write("  ✓ Excelente ajuste (R² ≥ 0.90)\n")
        elif best_metrics['r2'] >= 0.7:
            f.write("  ○ Bom ajuste (R² entre 0.70 e 0.90)\n")
        else:
            f.write("  ⚠ Ajuste moderado/baixo (R² < 0.70)\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"  ✓ Relatório salvo: {output_path}")


def print_ranking(df_results: pd.DataFrame, best_name: str):
    """
    Imprime ranking dos modelos no console.

    Args:
        df_results: DataFrame com resultados
        best_name: Nome do melhor modelo
    """
    print("\n" + "=" * 60)
    print("RANKING DOS MODELOS (por R²)")
    print("=" * 60)

    df_sorted = df_results.sort_values("r2", ascending=False)

    print(f"\n{'Rank':<6} {'Modelo':<20} {'R²':<10} {'MSE':<12} {'MAE':<12}")
    print("-" * 60)

    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        marker = "★ MELHOR" if row["model"] == best_name else ""
        print(f"{rank:<6} {row['model']:<20} {row['r2']:.4f}     {row['mse']:<12.2f} {row['mae']:<12.2f} {marker}")


def main(**kwargs) -> dict:
    """
    Função principal - Etapa 5: Validação e Avaliação.

    Returns:
        Dicionário com resultados da execução
    """
    print("=" * 60)
    print("ETAPA 5: VALIDAÇÃO E AVALIAÇÃO")
    print("(Conforme fluxos.drawio)")
    print("=" * 60)

    # Verificar arquivos de entrada
    test_data_file = Path("train_test_split.npz")
    models_dir = Path("models")

    if not test_data_file.exists():
        print(f"\n✗ Arquivo não encontrado: {test_data_file}")
        print("Execute a Etapa 4 primeiro (s04_modeling.py)")
        return {"status": "error", "message": "Test data not found"}

    if not models_dir.exists():
        print(f"\n✗ Diretório não encontrado: {models_dir}")
        print("Execute a Etapa 4 primeiro (s04_modeling.py)")
        return {"status": "error", "message": "Models directory not found"}

    # Carregar dados de teste
    print("\n[1/5] Carregando dados de teste (20%)...")
    X_test, y_test, feature_names = load_test_data(str(test_data_file))

    # Carregar modelos
    print("\n[2/5] Carregando modelos treinados...")
    models = load_models(str(models_dir))

    if not models:
        print("\n✗ Nenhum modelo encontrado")
        return {"status": "error", "message": "No models found"}

    # Avaliar e comparar modelos
    print("\n" + "-" * 40)
    print("TESTE NOS 20% DE DADOS")
    print("-" * 40)
    print("\n[3/5] Calculando métricas (R², MSE, MAE)...")
    df_results = compare_models(models, X_test, y_test)

    # Comparar desempenho
    print("\n" + "-" * 40)
    print("COMPARAÇÃO DE DESEMPENHO")
    print("-" * 40)
    print_ranking(df_results, "")

    # Selecionar melhor modelo
    print("\n[4/5] Selecionando melhor modelo...")
    best_name, best_model, best_metrics = select_best_model(df_results, models, criterion="r2")

    print(f"\n  ★ Melhor modelo: {best_name.upper()}")
    print(f"    R²:   {best_metrics['r2']:.4f}")
    print(f"    MSE:  {best_metrics['mse']:.2f}")
    print(f"    MAE:  {best_metrics['mae']:.2f}")

    # Imprimir ranking final
    print_ranking(df_results, best_name)

    # Salvar melhor modelo e relatório
    print("\n[5/5] Salvando resultados...")
    save_best_model(best_model, best_name, best_metrics, "best_model.joblib")
    generate_evaluation_report(df_results, best_name, best_metrics, "evaluation_report.txt")

    # Resumo
    print("\n" + "=" * 60)
    print("ETAPA 5 CONCLUÍDA")
    print("=" * 60)
    print(f"\nMelhor modelo: {best_name.upper()}")
    print(f"  R²:  {best_metrics['r2']:.4f}")
    print(f"  MSE: {best_metrics['mse']:.2f}")
    print(f"  MAE: {best_metrics['mae']:.2f}")
    print(f"\nArquivos gerados:")
    print(f"  - best_model.joblib")
    print(f"  - evaluation_report.txt")

    results = {
        "status": "success",
        "best_model": best_name,
        "metrics": best_metrics,
        "all_results": df_results.to_dict('records'),
    }

    return results


if __name__ == "__main__":
    main()
