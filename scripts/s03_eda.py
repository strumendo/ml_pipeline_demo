"""
S03 - Análise Exploratória de Dados (EDA)
=========================================
Etapa 3 do Pipeline conforme fluxos.drawio

O QUE FAZ:
- Cálculo de Estatísticas: Média, Desvio Padrão, Quartis
- Gráficos de Distribuição: Histogramas, Boxplots
- Análise de Correlação: Heatmaps, Dispersão
- Gera insights iniciais sobre os dados

FLUXO (fluxos.drawio):
Base para EDA → Estatísticas + Gráficos + Correlação → Insights iniciais → Base Pós-EDA

ENTRADA:
- data_preprocessed.csv (saída da Etapa 2)

SAÍDA:
- data_eda.csv: Dados prontos para modelagem
- eda_report.txt: Relatório com estatísticas e insights
- eda_plots/: Diretório com gráficos gerados
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Tentar importar bibliotecas de visualização
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend não-interativo para geração de arquivos
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠ Matplotlib/Seaborn não disponível. Gráficos serão pulados.")


def load_preprocessed_data(filepath: str = "data_preprocessed.csv") -> pd.DataFrame:
    """
    Carrega dados pré-processados da Etapa 2.

    Args:
        filepath: Caminho do arquivo CSV

    Returns:
        DataFrame com dados pré-processados
    """
    df = pd.read_csv(filepath)
    print(f"  Carregado: {len(df)} registros, {len(df.columns)} colunas")
    return df


def calculate_statistics(df: pd.DataFrame) -> dict:
    """
    Calcula estatísticas descritivas: Média, Desvio Padrão, Quartis.

    Args:
        df: DataFrame de entrada

    Returns:
        Dicionário com estatísticas
    """
    print("\n  Calculando estatísticas descritivas...")

    # Selecionar apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = {}

    for col in numeric_cols:
        col_stats = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "q1": df[col].quantile(0.25),
            "median": df[col].quantile(0.50),
            "q3": df[col].quantile(0.75),
            "max": df[col].max(),
            "nulls": df[col].isnull().sum(),
        }
        stats[col] = col_stats

    print(f"  ✓ Estatísticas calculadas para {len(numeric_cols)} colunas numéricas")

    return stats


def print_statistics_report(stats: dict) -> str:
    """
    Gera relatório formatado das estatísticas.

    Args:
        stats: Dicionário com estatísticas

    Returns:
        String com relatório formatado
    """
    report = []
    report.append("=" * 70)
    report.append("ESTATÍSTICAS DESCRITIVAS")
    report.append("=" * 70)

    for col, col_stats in stats.items():
        report.append(f"\n{col}:")
        report.append(f"  Média:    {col_stats['mean']:.4f}")
        report.append(f"  Desvio:   {col_stats['std']:.4f}")
        report.append(f"  Mínimo:   {col_stats['min']:.4f}")
        report.append(f"  Q1 (25%): {col_stats['q1']:.4f}")
        report.append(f"  Mediana:  {col_stats['median']:.4f}")
        report.append(f"  Q3 (75%): {col_stats['q3']:.4f}")
        report.append(f"  Máximo:   {col_stats['max']:.4f}")

    return "\n".join(report)


def generate_distribution_plots(df: pd.DataFrame, output_dir: Path) -> list:
    """
    Gera gráficos de distribuição: Histogramas e Boxplots.

    Args:
        df: DataFrame de entrada
        output_dir: Diretório para salvar gráficos

    Returns:
        Lista de arquivos gerados
    """
    if not HAS_PLOTTING:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    files_generated = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Limitar a 10 colunas mais importantes
    if len(numeric_cols) > 10:
        # Priorizar colunas não-dummies (sem prefixo de encoding)
        priority_cols = [c for c in numeric_cols if not any(c.startswith(p) for p in ["Cod_", "Equipamento_", "Descricao_"])]
        numeric_cols = priority_cols[:10] if priority_cols else numeric_cols[:10]

    # Histogramas
    print("\n  Gerando histogramas...")
    fig, axes = plt.subplots(
        nrows=(len(numeric_cols) + 2) // 3,
        ncols=3,
        figsize=(15, 4 * ((len(numeric_cols) + 2) // 3))
    )
    axes = axes.flatten() if len(numeric_cols) > 1 else [axes]

    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{col}', fontsize=10)
            axes[i].set_xlabel('')

    # Remover eixos vazios
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    hist_path = output_dir / "histogramas.png"
    plt.savefig(hist_path, dpi=100)
    plt.close()
    files_generated.append(str(hist_path))
    print(f"  ✓ Histogramas salvos: {hist_path}")

    # Boxplots
    print("\n  Gerando boxplots...")
    fig, axes = plt.subplots(
        nrows=(len(numeric_cols) + 2) // 3,
        ncols=3,
        figsize=(15, 4 * ((len(numeric_cols) + 2) // 3))
    )
    axes = axes.flatten() if len(numeric_cols) > 1 else [axes]

    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(f'{col}', fontsize=10)

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    box_path = output_dir / "boxplots.png"
    plt.savefig(box_path, dpi=100)
    plt.close()
    files_generated.append(str(box_path))
    print(f"  ✓ Boxplots salvos: {box_path}")

    return files_generated


def generate_correlation_analysis(df: pd.DataFrame, output_dir: Path) -> tuple:
    """
    Gera análise de correlação: Heatmap e Gráficos de Dispersão.

    Args:
        df: DataFrame de entrada
        output_dir: Diretório para salvar gráficos

    Returns:
        Tupla (matriz de correlação, lista de arquivos)
    """
    files_generated = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calcular matriz de correlação
    print("\n  Calculando matriz de correlação...")
    corr_matrix = df[numeric_cols].corr()

    # Identificar correlações fortes com o target
    if "Manutencao" in numeric_cols:
        target_corr = corr_matrix["Manutencao"].drop("Manutencao").sort_values(key=abs, ascending=False)
        print("\n  Top 10 correlações com 'Manutencao':")
        for col, corr in target_corr.head(10).items():
            print(f"    {col}: {corr:.4f}")

    if not HAS_PLOTTING:
        return corr_matrix, files_generated

    output_dir.mkdir(parents=True, exist_ok=True)

    # Selecionar top 15 colunas para o heatmap (evitar matriz muito grande)
    if len(numeric_cols) > 15:
        if "Manutencao" in numeric_cols:
            # Pegar colunas mais correlacionadas com target
            top_cols = ["Manutencao"] + list(target_corr.head(14).index)
        else:
            top_cols = numeric_cols[:15]
    else:
        top_cols = numeric_cols

    # Heatmap
    print("\n  Gerando heatmap de correlação...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df[top_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8}
    )
    plt.title("Matriz de Correlação", fontsize=14)
    plt.tight_layout()
    heatmap_path = output_dir / "heatmap_correlacao.png"
    plt.savefig(heatmap_path, dpi=100)
    plt.close()
    files_generated.append(str(heatmap_path))
    print(f"  ✓ Heatmap salvo: {heatmap_path}")

    # Gráficos de dispersão (target vs top features)
    if "Manutencao" in numeric_cols:
        print("\n  Gerando gráficos de dispersão...")
        top_features = list(target_corr.head(4).index)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, feat in enumerate(top_features):
            axes[i].scatter(df[feat], df["Manutencao"], alpha=0.5, s=10)
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel("Manutencao")
            axes[i].set_title(f"Correlação: {corr_matrix.loc[feat, 'Manutencao']:.4f}")

        plt.tight_layout()
        scatter_path = output_dir / "dispersao_target.png"
        plt.savefig(scatter_path, dpi=100)
        plt.close()
        files_generated.append(str(scatter_path))
        print(f"  ✓ Dispersão salvo: {scatter_path}")

    return corr_matrix, files_generated


def generate_insights(df: pd.DataFrame, stats: dict, corr_matrix: pd.DataFrame) -> list:
    """
    Gera insights automáticos sobre os dados.

    Args:
        df: DataFrame de entrada
        stats: Estatísticas descritivas
        corr_matrix: Matriz de correlação

    Returns:
        Lista de insights
    """
    insights = []

    # Insight 1: Tamanho do dataset
    insights.append(f"Dataset contém {len(df)} registros e {len(df.columns)} features.")

    # Insight 2: Variável target
    if "Manutencao" in df.columns:
        maint_stats = stats.get("Manutencao", {})
        if maint_stats:
            insights.append(
                f"Variável target 'Manutencao': média de {maint_stats['mean']:.1f} dias, "
                f"variando de {maint_stats['min']:.0f} a {maint_stats['max']:.0f} dias."
            )

    # Insight 3: Correlações fortes
    if "Manutencao" in corr_matrix.columns:
        target_corr = corr_matrix["Manutencao"].drop("Manutencao", errors='ignore')
        strong_corr = target_corr[abs(target_corr) > 0.5]
        if len(strong_corr) > 0:
            insights.append(
                f"Encontradas {len(strong_corr)} features com correlação forte (>0.5) com o target: "
                f"{', '.join(strong_corr.index[:5].tolist())}"
            )

    # Insight 4: Outliers (usando IQR)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_cols = []
    for col in numeric_cols:
        if col in stats:
            q1, q3 = stats[col]['q1'], stats[col]['q3']
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > len(df) * 0.05:  # Mais de 5% de outliers
                outlier_cols.append(col)

    if outlier_cols:
        insights.append(
            f"Atenção: {len(outlier_cols)} colunas com >5% de outliers: "
            f"{', '.join(outlier_cols[:5])}"
        )

    # Insight 5: Balanceamento (se houver categorias)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols[:3]:  # Limitar análise
        value_counts = df[col].value_counts(normalize=True)
        if value_counts.max() > 0.8:
            insights.append(f"Coluna '{col}' desbalanceada: {value_counts.idxmax()} representa {value_counts.max()*100:.1f}%")

    return insights


def save_eda_report(stats: dict, insights: list, output_path: str = "eda_report.txt"):
    """
    Salva relatório completo de EDA.

    Args:
        stats: Estatísticas descritivas
        insights: Lista de insights
        output_path: Caminho do arquivo de saída
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE ANÁLISE EXPLORATÓRIA (EDA)\n")
        f.write("=" * 70 + "\n\n")

        # Insights
        f.write("INSIGHTS PRINCIPAIS\n")
        f.write("-" * 40 + "\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")
        f.write("\n")

        # Estatísticas
        f.write(print_statistics_report(stats))
        f.write("\n")

    print(f"  ✓ Relatório salvo: {output_path}")


def main(**kwargs) -> dict:
    """
    Função principal - Etapa 3: Análise Exploratória.

    Returns:
        Dicionário com resultados da execução
    """
    print("=" * 60)
    print("ETAPA 3: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("(Conforme fluxos.drawio)")
    print("=" * 60)

    # Verificar arquivo de entrada
    input_file = Path("data_preprocessed.csv")
    if not input_file.exists():
        print(f"\n✗ Arquivo não encontrado: {input_file}")
        print("Execute a Etapa 2 primeiro (s02_preprocessing.py)")
        return {"status": "error", "message": "Input file not found"}

    # Carregar dados
    print("\n[1/5] Carregando dados pré-processados...")
    df = load_preprocessed_data(str(input_file))

    # Calcular estatísticas
    print("\n" + "-" * 40)
    print("CÁLCULO DE ESTATÍSTICAS")
    print("-" * 40)
    print("\n[2/5] Média, Desvio Padrão, Quartis...")
    stats = calculate_statistics(df)

    # Gerar gráficos
    output_dir = Path("eda_plots")
    files_generated = []

    print("\n" + "-" * 40)
    print("GRÁFICOS DE DISTRIBUIÇÃO")
    print("-" * 40)
    print("\n[3/5] Histogramas e Boxplots...")
    files_generated.extend(generate_distribution_plots(df, output_dir))

    print("\n" + "-" * 40)
    print("ANÁLISE DE CORRELAÇÃO")
    print("-" * 40)
    print("\n[4/5] Heatmaps e Dispersão...")
    corr_matrix, corr_files = generate_correlation_analysis(df, output_dir)
    files_generated.extend(corr_files)

    # Gerar insights
    print("\n" + "-" * 40)
    print("INSIGHTS")
    print("-" * 40)
    print("\n[5/5] Gerando insights automáticos...")
    insights = generate_insights(df, stats, corr_matrix)

    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")

    # Salvar relatório
    save_eda_report(stats, insights, "eda_report.txt")

    # Salvar dados para próxima etapa (mesmo arquivo, apenas confirma integridade)
    output_file = Path("data_eda.csv")
    df.to_csv(output_file, index=False)

    # Resumo
    print("\n" + "=" * 60)
    print("ETAPA 3 CONCLUÍDA")
    print("=" * 60)
    print(f"\nArquivos gerados:")
    print(f"  - data_eda.csv (dados para modelagem)")
    print(f"  - eda_report.txt (relatório estatístico)")
    if files_generated:
        print(f"  - {len(files_generated)} gráficos em eda_plots/")

    results = {
        "status": "success",
        "registros": len(df),
        "colunas": len(df.columns),
        "insights": insights,
        "plots_generated": len(files_generated),
        "output_file": str(output_file),
    }

    return results


if __name__ == "__main__":
    main()
