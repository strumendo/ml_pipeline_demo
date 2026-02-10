"""
S04 - Modelagem e Treinamento
=============================
Etapa 4 do Pipeline conforme fluxos.drawio

O QUE FAZ:
- Divide dados em Treino/Teste (80%/20%)
- Treina 4 algoritmos em paralelo:
  1. Regressão Linear
  2. Decision Tree
  3. Random Forest
  4. XGBoost
- Salva os 4 modelos treinados

FLUXO (fluxos.drawio):
Base Pós-EDA → Dividir Treino/Teste (80%/20%) → [4 Algoritmos] → 4 Modelos Treinados

ENTRADA:
- data_eda.csv (saída da Etapa 3)

SAÍDA:
- models/model_linear.joblib
- models/model_decision_tree.joblib
- models/model_random_forest.joblib
- models/model_xgboost.joblib
- train_test_split.npz (dados de treino/teste)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost não disponível. Usando GradientBoostingRegressor.")
    from sklearn.ensemble import GradientBoostingRegressor


def load_eda_data(filepath: str = "data_eda.csv") -> pd.DataFrame:
    """
    Carrega dados da etapa de EDA.

    Args:
        filepath: Caminho do arquivo CSV

    Returns:
        DataFrame com dados
    """
    df = pd.read_csv(filepath)
    print(f"  Carregado: {len(df)} registros, {len(df.columns)} colunas")
    return df


def prepare_features_target(df: pd.DataFrame, target_col: str = "Manutencao") -> tuple:
    """
    Separa features e variável target.

    Args:
        df: DataFrame de entrada
        target_col: Nome da coluna target

    Returns:
        Tupla (X, y, feature_names)
    """
    if target_col not in df.columns:
        raise ValueError(f"Coluna target '{target_col}' não encontrada no DataFrame")

    # Remover colunas não-numéricas e target
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col != target_col]

    # Remover colunas que são totalmente NaN ou "Unnamed"
    cols_to_remove = [col for col in feature_cols
                      if col.startswith("Unnamed") or df[col].isna().all()]
    if cols_to_remove:
        feature_cols = [col for col in feature_cols if col not in cols_to_remove]
        print(f"  ⚠ Removidas {len(cols_to_remove)} colunas vazias/unnamed: {cols_to_remove}")

    # CORREÇÃO: Remover features com data leakage
    # intervalo_manutencao correlaciona ~1.0 com target (causa overfitting)
    # Estas features vazam informação do target para o modelo
    leaky_features = [
        "intervalo_manutencao",  # Intervalo fixo por equipamento = target direto
    ]
    removed_features = []
    for leaky in leaky_features:
        if leaky in feature_cols:
            feature_cols.remove(leaky)
            removed_features.append(leaky)

    if removed_features:
        print(f"  ⚠ Removidas {len(removed_features)} features com data leakage: {removed_features}")

    X = df[feature_cols].values
    y = df[target_col].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: {target_col}")

    return X, y, feature_cols


def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Divide dados em treino e teste (80%/20% conforme diagrama).

    Args:
        X: Features
        y: Target
        test_size: Proporção de teste (0.2 = 20%)
        random_state: Seed para reprodutibilidade

    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\n  Divisão Treino/Teste ({int((1-test_size)*100)}%/{int(test_size*100)}%):")
    print(f"    Treino: {len(X_train)} amostras")
    print(f"    Teste:  {len(X_test)} amostras")

    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Treina modelo de Regressão Linear.

    Algoritmo 1 conforme fluxos.drawio.

    Args:
        X_train: Features de treino
        y_train: Target de treino

    Returns:
        Modelo treinado
    """
    print("\n  [Algoritmo 1] Regressão Linear...")

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("    ✓ Modelo treinado")

    return model


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> DecisionTreeRegressor:
    """
    Treina modelo Decision Tree.

    Algoritmo 2 conforme fluxos.drawio.

    Args:
        X_train: Features de treino
        y_train: Target de treino
        random_state: Seed para reprodutibilidade

    Returns:
        Modelo treinado
    """
    print("\n  [Algoritmo 2] Decision Tree...")

    # Hiperparâmetros ajustados para evitar overfitting
    model = DecisionTreeRegressor(
        max_depth=6,            # Reduzido de 10 para evitar overfitting
        min_samples_split=10,   # Aumentado de 5
        min_samples_leaf=10,    # Aumentado de 2 para evitar memorização
        random_state=random_state
    )
    model.fit(X_train, y_train)

    print("    ✓ Modelo treinado")

    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> RandomForestRegressor:
    """
    Treina modelo Random Forest.

    Algoritmo 3 conforme fluxos.drawio.

    Args:
        X_train: Features de treino
        y_train: Target de treino
        random_state: Seed para reprodutibilidade

    Returns:
        Modelo treinado
    """
    print("\n  [Algoritmo 3] Random Forest...")

    # Hiperparâmetros ajustados para evitar overfitting
    # max_depth reduzido de 15 para 8 (evita memorização)
    # min_samples_leaf aumentado de 2 para 10 (generalização melhor)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,           # Reduzido de 15 para evitar overfitting
        min_samples_split=10,  # Aumentado de 5
        min_samples_leaf=10,   # Aumentado de 2 para evitar memorização
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("    ✓ Modelo treinado")

    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """
    Treina modelo XGBoost.

    Algoritmo 4 conforme fluxos.drawio.

    Args:
        X_train: Features de treino
        y_train: Target de treino
        random_state: Seed para reprodutibilidade

    Returns:
        Modelo treinado
    """
    print("\n  [Algoritmo 4] XGBoost...")

    if HAS_XGBOOST:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        # Fallback para GradientBoosting
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state
        )

    model.fit(X_train, y_train)

    print("    ✓ Modelo treinado")

    return model


def save_models(models: dict, output_dir: Path):
    """
    Salva todos os modelos treinados.

    Args:
        models: Dicionário com modelos {nome: modelo}
        output_dir: Diretório de saída
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        filepath = output_dir / f"model_{name}.joblib"
        joblib.dump(model, filepath)
        print(f"  ✓ Salvo: {filepath}")


def save_train_test_data(X_train, X_test, y_train, y_test, feature_names, output_path: str = "train_test_split.npz"):
    """
    Salva dados de treino/teste para etapa de avaliação.

    Args:
        X_train, X_test, y_train, y_test: Dados divididos
        feature_names: Nomes das features
        output_path: Caminho do arquivo de saída
    """
    np.savez(
        output_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=np.array(feature_names)
    )
    print(f"  ✓ Dados treino/teste salvos: {output_path}")


def main(**kwargs) -> dict:
    """
    Função principal - Etapa 4: Modelagem e Treinamento.

    Returns:
        Dicionário com resultados da execução
    """
    print("=" * 60)
    print("ETAPA 4: MODELAGEM E TREINAMENTO")
    print("(Conforme fluxos.drawio)")
    print("=" * 60)

    # Verificar arquivo de entrada
    input_file = Path("data_eda.csv")
    if not input_file.exists():
        print(f"\n✗ Arquivo não encontrado: {input_file}")
        print("Execute a Etapa 3 primeiro (s03_eda.py)")
        return {"status": "error", "message": "Input file not found"}

    # Carregar dados
    print("\n[1/7] Carregando dados pós-EDA...")
    df = load_eda_data(str(input_file))

    # Preparar features e target
    print("\n[2/7] Preparando features e target...")
    X, y, feature_names = prepare_features_target(df, "Manutencao")

    # Dividir treino/teste (80%/20%)
    print("\n" + "-" * 40)
    print("DIVISÃO TREINO/TESTE")
    print("-" * 40)
    print("\n[3/7] Dividindo dados (80% treino / 20% teste)...")
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)

    # Salvar dados de treino/teste
    save_train_test_data(X_train, X_test, y_train, y_test, feature_names)

    # Treinar os 4 algoritmos
    print("\n" + "-" * 40)
    print("TREINAMENTO DOS MODELOS")
    print("-" * 40)

    models = {}

    print("\n[4/7] Treinando Regressão Linear...")
    models["linear"] = train_linear_regression(X_train, y_train)

    print("\n[5/7] Treinando Decision Tree...")
    models["decision_tree"] = train_decision_tree(X_train, y_train)

    print("\n[6/7] Treinando Random Forest...")
    models["random_forest"] = train_random_forest(X_train, y_train)

    print("\n[7/7] Treinando XGBoost...")
    models["xgboost"] = train_xgboost(X_train, y_train)

    # Salvar modelos
    print("\n" + "-" * 40)
    print("SALVANDO MODELOS")
    print("-" * 40)
    output_dir = Path("models")
    save_models(models, output_dir)

    # Resumo
    print("\n" + "=" * 60)
    print("ETAPA 4 CONCLUÍDA")
    print("=" * 60)
    print(f"\nModelos treinados: {len(models)}")
    for name in models.keys():
        print(f"  - {name}")
    print(f"\nArquivos gerados:")
    print(f"  - models/*.joblib (4 modelos)")
    print(f"  - train_test_split.npz (dados treino/teste)")

    results = {
        "status": "success",
        "models_trained": list(models.keys()),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": len(feature_names),
        "output_dir": str(output_dir),
    }

    return results


if __name__ == "__main__":
    main()
