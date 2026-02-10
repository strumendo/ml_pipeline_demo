"""
S03b - Análises Avançadas de EDA
================================
Complemento da Etapa 3 - Análises baseadas nos notebooks exploratórios

O QUE FAZ:
- Gráficos avançados baseados em exploratory_analise/
- Análise por equipamento (correlation_matrix_cilindros)
- Consumo de massa vs quantidade produzida
- Ano de construção vs consumo
- Matriz de confusão para classificação de urgência

ENTRADA:
- data_eda.csv (saída da Etapa 3)
- Dados dos notebooks exploratórios como referência

SAÍDA:
- eda_plots/correlation_matrix_full.png
- eda_plots/consumo_vs_producao.png
- eda_plots/analise_temporal.png
- eda_plots/matriz_confusao_urgencia.png
- eda_plots/scatter_plots_features.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')  # Backend não-interativo para geração de arquivos
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠ Matplotlib/Seaborn não disponível.")


def generate_full_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Gera matriz de correlação completa similar a correlation_matrix_cilindros.png.

    Args:
        df: DataFrame de entrada
        output_dir: Diretório de saída

    Returns:
        Caminho do arquivo gerado
    """
    if not HAS_PLOTTING:
        return None

    print("\n  Gerando matriz de correlação completa...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Limitar a 20 colunas mais relevantes para visualização
    if len(numeric_cols) > 20:
        # Priorizar colunas sem encoding
        priority = [c for c in numeric_cols if not any(p in c.lower() for p in ['cod_', 'equipamento_', 'produto_'])]
        if len(priority) >= 10:
            numeric_cols = priority[:20]
        else:
            numeric_cols = numeric_cols[:20]

    corr = df[numeric_cols].corr()

    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Matriz de Correlação Completa", fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "correlation_matrix_full.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Matriz de correlação salva: {output_path}")
    return str(output_path)


def generate_consumo_vs_producao(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Gera gráfico de consumo de massa vs quantidade produzida.
    Similar a consumo_massa_vs_qtd_produzida.png.

    Args:
        df: DataFrame de entrada
        output_dir: Diretório de saída

    Returns:
        Caminho do arquivo gerado
    """
    if not HAS_PLOTTING:
        return None

    print("\n  Gerando gráfico Consumo vs Produção...")

    # Identificar colunas relevantes
    consumo_col = None
    producao_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'consumo' in col_lower and 'massa' in col_lower:
            consumo_col = col
        elif 'produzida' in col_lower or 'producao' in col_lower:
            producao_col = col

    if not consumo_col or not producao_col:
        # Tentar alternativas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            consumo_col = numeric_cols[0]
            producao_col = numeric_cols[1]
        else:
            print("  ⚠ Colunas de consumo/produção não encontradas")
            return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    axes[0].scatter(df[producao_col], df[consumo_col], alpha=0.5, s=30, c='#F77F00')
    axes[0].set_xlabel(producao_col, fontsize=11)
    axes[0].set_ylabel(consumo_col, fontsize=11)
    axes[0].set_title("Consumo de Massa vs Quantidade Produzida", fontsize=12, fontweight='bold')

    # Adicionar linha de tendência
    if len(df) > 10:
        z = np.polyfit(df[producao_col].dropna(), df[consumo_col].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[producao_col].min(), df[producao_col].max(), 100)
        axes[0].plot(x_line, p(x_line), "r--", alpha=0.8, label=f"Tendência linear")
        axes[0].legend()

    # Boxplot por faixas de produção
    df_temp = df.copy()
    df_temp['Faixa_Producao'] = pd.qcut(df[producao_col], q=4, labels=['Baixa', 'Média-Baixa', 'Média-Alta', 'Alta'])
    df_temp.boxplot(column=consumo_col, by='Faixa_Producao', ax=axes[1])
    axes[1].set_xlabel("Faixa de Produção", fontsize=11)
    axes[1].set_ylabel(consumo_col, fontsize=11)
    axes[1].set_title("Consumo por Faixa de Produção", fontsize=12, fontweight='bold')
    plt.suptitle('')  # Remove título automático do boxplot

    plt.tight_layout()

    output_path = output_dir / "consumo_vs_producao.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráfico consumo vs produção salvo: {output_path}")
    return str(output_path)


def generate_temporal_analysis(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Gera análise temporal similar a ano_construcao_vs_consumo_massa.png.

    Args:
        df: DataFrame de entrada
        output_dir: Diretório de saída

    Returns:
        Caminho do arquivo gerado
    """
    if not HAS_PLOTTING:
        return None

    print("\n  Gerando análise temporal...")

    # Identificar colunas de data/tempo
    date_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in ['data', 'date', 'ano', 'year', 'manutencao', 'dias']):
            date_cols.append(col)

    if not date_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            date_cols = numeric_cols[:2]
        else:
            print("  ⚠ Colunas temporais não encontradas")
            return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Gráfico 1: Histograma da variável principal
    target_col = 'Manutencao' if 'Manutencao' in df.columns else date_cols[0]
    axes[0, 0].hist(df[target_col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='#F77F00')
    axes[0, 0].set_xlabel(target_col, fontsize=11)
    axes[0, 0].set_ylabel("Frequência", fontsize=11)
    axes[0, 0].set_title(f"Distribuição de {target_col}", fontsize=12, fontweight='bold')
    axes[0, 0].axvline(df[target_col].mean(), color='red', linestyle='--', label=f'Média: {df[target_col].mean():.1f}')
    axes[0, 0].legend()

    # Gráfico 2: Boxplot da variável principal
    axes[0, 1].boxplot(df[target_col].dropna(), vert=True)
    axes[0, 1].set_ylabel(target_col, fontsize=11)
    axes[0, 1].set_title(f"Boxplot de {target_col}", fontsize=12, fontweight='bold')

    # Gráfico 3: Tendência acumulada
    if len(df) > 10:
        sorted_df = df[target_col].dropna().sort_values()
        axes[1, 0].plot(range(len(sorted_df)), sorted_df.values, color='#F77F00', linewidth=1.5)
        axes[1, 0].fill_between(range(len(sorted_df)), sorted_df.values, alpha=0.3, color='#F77F00')
        axes[1, 0].set_xlabel("Índice Ordenado", fontsize=11)
        axes[1, 0].set_ylabel(target_col, fontsize=11)
        axes[1, 0].set_title("Distribuição Acumulada", fontsize=12, fontweight='bold')

    # Gráfico 4: Quartis
    quartis = df[target_col].quantile([0.25, 0.5, 0.75]).values
    labels = ['Q1 (25%)', 'Mediana (50%)', 'Q3 (75%)']
    colors = ['#3498db', '#F77F00', '#e74c3c']
    axes[1, 1].bar(labels, quartis, color=colors, edgecolor='black')
    axes[1, 1].set_ylabel(target_col, fontsize=11)
    axes[1, 1].set_title("Quartis", fontsize=12, fontweight='bold')
    for i, v in enumerate(quartis):
        axes[1, 1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / "analise_temporal.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Análise temporal salva: {output_path}")
    return str(output_path)


def generate_urgency_matrix(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Gera matriz de confusão para classificação de urgência.
    Similar às matrizes de confusão dos notebooks de modelagem.

    Args:
        df: DataFrame de entrada
        output_dir: Diretório de saída

    Returns:
        Caminho do arquivo gerado
    """
    if not HAS_PLOTTING:
        return None

    print("\n  Gerando matriz de urgência...")

    # Identificar coluna de manutenção
    manut_col = None
    for col in df.columns:
        if 'manutencao' in col.lower() or 'manut' in col.lower():
            manut_col = col
            break

    if not manut_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            manut_col = numeric_cols[0]
        else:
            print("  ⚠ Coluna de manutenção não encontrada")
            return None

    # Classificar em urgência (Curto, Médio, Longo)
    try:
        # Usar quartis para definir faixas
        q1 = df[manut_col].quantile(0.33)
        q2 = df[manut_col].quantile(0.66)

        def classify_urgency(val):
            if pd.isna(val):
                return 'Indefinido'
            elif val <= q1:
                return 'Curto'
            elif val <= q2:
                return 'Médio'
            else:
                return 'Longo'

        df_temp = df.copy()
        df_temp['Urgencia'] = df_temp[manut_col].apply(classify_urgency)

        # Criar matriz de contagem
        urgency_counts = df_temp['Urgencia'].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Gráfico de barras
        colors = {'Curto': '#e74c3c', 'Médio': '#f39c12', 'Longo': '#27ae60', 'Indefinido': '#95a5a6'}
        bar_colors = [colors.get(u, '#3498db') for u in urgency_counts.index]

        axes[0].bar(urgency_counts.index, urgency_counts.values, color=bar_colors, edgecolor='black')
        axes[0].set_xlabel("Nível de Urgência", fontsize=11)
        axes[0].set_ylabel("Quantidade", fontsize=11)
        axes[0].set_title("Distribuição de Urgência de Manutenção", fontsize=12, fontweight='bold')

        for i, (label, val) in enumerate(zip(urgency_counts.index, urgency_counts.values)):
            axes[0].text(i, val + 0.5, str(val), ha='center', fontweight='bold')

        # Gráfico de pizza
        axes[1].pie(
            urgency_counts.values,
            labels=urgency_counts.index,
            autopct='%1.1f%%',
            colors=bar_colors,
            explode=[0.05 if u == 'Curto' else 0 for u in urgency_counts.index],
            shadow=True
        )
        axes[1].set_title("Proporção de Urgência", fontsize=12, fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / "matriz_urgencia.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Matriz de urgência salva: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"  ⚠ Erro ao gerar matriz de urgência: {e}")
        return None


def get_equipment_from_onehot(df: pd.DataFrame) -> pd.Series:
    """
    Extrai o nome do equipamento das colunas one-hot encoded.

    Args:
        df: DataFrame com colunas Equipamento_IJ_XXX

    Returns:
        Series com nome do equipamento para cada registro
    """
    equip_cols = [c for c in df.columns if c.startswith('Equipamento_')]

    if not equip_cols:
        return pd.Series(['Desconhecido'] * len(df), index=df.index)

    # Para cada registro, encontrar qual coluna é True
    equipamentos = []
    for idx in range(len(df)):
        equip = 'Outro'
        for col in equip_cols:
            if df.iloc[idx][col] == True or df.iloc[idx][col] == 1:
                equip = col.replace('Equipamento_', '').replace('_', '-')
                break
        equipamentos.append(equip)

    return pd.Series(equipamentos, index=df.index)


def generate_scatter_plots(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Gera múltiplos scatter plots das principais features, coloridos por equipamento.

    Args:
        df: DataFrame de entrada
        output_dir: Diretório de saída

    Returns:
        Caminho do arquivo gerado
    """
    if not HAS_PLOTTING:
        return None

    print("\n  Gerando scatter plots de features (colorido por equipamento)...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Selecionar top 6 features (excluindo dummies)
    features = [c for c in numeric_cols if not any(p in c.lower() for p in ['cod_', 'equipamento_', 'produto_', 'unnamed'])][:6]

    if len(features) < 3:
        features = numeric_cols[:6]

    if len(features) < 2:
        print("  ⚠ Features insuficientes para scatter plots")
        return None

    # Identificar equipamento de cada registro
    equipamentos = get_equipment_from_onehot(df)
    unique_equips = sorted(equipamentos.unique())

    # Criar paleta de cores para equipamentos
    n_equips = len(unique_equips)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_equips)))
    if n_equips > 20:
        colors = plt.cm.viridis(np.linspace(0, 1, n_equips))
    color_map = {eq: colors[i % len(colors)] for i, eq in enumerate(unique_equips)}

    # Criar grid de scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    plot_idx = 0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if plot_idx >= 6:
                break

            # Calcular correlação
            corr = df[[features[i], features[j]]].corr().iloc[0, 1]

            # Plotar cada equipamento com cor diferente
            for equip in unique_equips:
                mask = equipamentos == equip
                if mask.sum() > 0:
                    axes[plot_idx].scatter(
                        df.loc[mask, features[i]],
                        df.loc[mask, features[j]],
                        alpha=0.5,
                        s=15,
                        c=[color_map[equip]],
                        label=equip if plot_idx == 0 else None  # Legenda só no primeiro
                    )

            axes[plot_idx].set_xlabel(features[i][:25], fontsize=9)
            axes[plot_idx].set_ylabel(features[j][:25], fontsize=9)
            axes[plot_idx].set_title(f"Corr: {corr:.3f}", fontsize=10, fontweight='bold')

            plot_idx += 1

    # Ocultar eixos não utilizados
    for idx in range(plot_idx, 6):
        axes[idx].set_visible(False)

    # Adicionar legenda global
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5),
                   title='Equipamento', fontsize=7, title_fontsize=9, ncol=1)

    plt.suptitle("Scatter Plots das Principais Features por Equipamento", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / "scatter_plots_features.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Scatter plots salvos: {output_path}")
    print(f"    Equipamentos identificados: {len(unique_equips)}")
    return str(output_path)


def generate_equipment_summary_plot(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Gera gráfico resumo por equipamento (produção, refugo, manutenção).

    Args:
        df: DataFrame de entrada
        output_dir: Diretório de saída

    Returns:
        Caminho do arquivo gerado
    """
    if not HAS_PLOTTING:
        return None

    print("\n  Gerando gráfico resumo por equipamento...")

    # Identificar equipamento de cada registro
    equipamentos = get_equipment_from_onehot(df)

    # Agregar métricas por equipamento
    df_temp = df.copy()
    df_temp['Equipamento'] = equipamentos

    # Calcular métricas agregadas
    agg_data = df_temp.groupby('Equipamento').agg({
        'Qtd_Produzida': 'sum',
        'Qtd_Refugada': 'sum',
        'Manutencao': 'mean'
    }).reset_index()

    agg_data['Taxa_Refugo'] = (agg_data['Qtd_Refugada'] / agg_data['Qtd_Produzida'] * 100).fillna(0)
    agg_data = agg_data.sort_values('Qtd_Produzida', ascending=True)

    # Criar figura com 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    # 1. Produção Total por Equipamento
    colors_prod = plt.cm.Blues(np.linspace(0.3, 0.9, len(agg_data)))
    bars1 = axes[0].barh(agg_data['Equipamento'], agg_data['Qtd_Produzida'] / 1000, color=colors_prod)
    axes[0].set_xlabel('Produção Total (mil peças)', fontsize=10)
    axes[0].set_title('Produção Total por Equipamento', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='y', labelsize=8)

    # Adicionar valores nas barras
    for bar, val in zip(bars1, agg_data['Qtd_Produzida'] / 1000):
        axes[0].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                     f'{val:.0f}k', va='center', fontsize=7)

    # 2. Taxa de Refugo por Equipamento
    colors_ref = plt.cm.Reds(np.linspace(0.3, 0.9, len(agg_data)))
    agg_sorted_ref = agg_data.sort_values('Taxa_Refugo', ascending=True)
    bars2 = axes[1].barh(agg_sorted_ref['Equipamento'], agg_sorted_ref['Taxa_Refugo'], color=colors_ref)
    axes[1].set_xlabel('Taxa de Refugo (%)', fontsize=10)
    axes[1].set_title('Taxa de Refugo por Equipamento', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='y', labelsize=8)
    axes[1].axvline(x=agg_data['Taxa_Refugo'].mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {agg_data["Taxa_Refugo"].mean():.1f}%')
    axes[1].legend(fontsize=8)

    # Adicionar valores nas barras
    for bar, val in zip(bars2, agg_sorted_ref['Taxa_Refugo']):
        axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}%', va='center', fontsize=7)

    # 3. Média de Dias até Manutenção por Equipamento
    colors_man = plt.cm.Greens(np.linspace(0.3, 0.9, len(agg_data)))
    agg_sorted_man = agg_data.sort_values('Manutencao', ascending=True)
    bars3 = axes[2].barh(agg_sorted_man['Equipamento'], agg_sorted_man['Manutencao'], color=colors_man)
    axes[2].set_xlabel('Média de Dias até Manutenção', fontsize=10)
    axes[2].set_title('Dias até Manutenção por Equipamento', fontsize=12, fontweight='bold')
    axes[2].tick_params(axis='y', labelsize=8)
    axes[2].axvline(x=agg_data['Manutencao'].mean(), color='green', linestyle='--', linewidth=2, label=f'Média: {agg_data["Manutencao"].mean():.0f} dias')
    axes[2].legend(fontsize=8)

    # Adicionar valores nas barras
    for bar, val in zip(bars3, agg_sorted_man['Manutencao']):
        axes[2].text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                     f'{val:.0f}d', va='center', fontsize=7)

    plt.suptitle(f"Resumo por Equipamento ({len(agg_data)} equipamentos)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "resumo_equipamentos.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráfico resumo salvos: {output_path}")
    print(f"    Total de equipamentos: {len(agg_data)}")
    return str(output_path)


def main(**kwargs) -> dict:
    """
    Função principal - Análises Avançadas de EDA.

    Returns:
        Dicionário com resultados
    """
    print("=" * 60)
    print("ETAPA 3b: ANÁLISES AVANÇADAS DE EDA")
    print("(Baseado em exploratory_analise/ e exploratory_dado_proposto/)")
    print("=" * 60)

    # Verificar arquivo de entrada
    input_file = Path("data_eda.csv")
    if not input_file.exists():
        print(f"\n✗ Arquivo não encontrado: {input_file}")
        print("Execute a Etapa 3 primeiro (s03_eda.py)")
        return {"status": "error", "message": "Input file not found"}

    # Carregar dados
    print("\n[1/7] Carregando dados...")
    df = pd.read_csv(str(input_file))
    print(f"  Carregado: {len(df)} registros, {len(df.columns)} colunas")

    output_dir = Path("eda_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    files_generated = []

    # Gerar gráficos avançados
    print("\n" + "-" * 40)
    print("GERANDO ANÁLISES AVANÇADAS")
    print("-" * 40)

    print("\n[2/7] Matriz de correlação completa...")
    path = generate_full_correlation_matrix(df, output_dir)
    if path:
        files_generated.append(path)

    print("\n[3/7] Consumo vs Produção...")
    path = generate_consumo_vs_producao(df, output_dir)
    if path:
        files_generated.append(path)

    print("\n[4/7] Análise temporal...")
    path = generate_temporal_analysis(df, output_dir)
    if path:
        files_generated.append(path)

    print("\n[5/7] Matriz de urgência...")
    path = generate_urgency_matrix(df, output_dir)
    if path:
        files_generated.append(path)

    print("\n[6/7] Scatter plots de features (por equipamento)...")
    path = generate_scatter_plots(df, output_dir)
    if path:
        files_generated.append(path)

    print("\n[7/7] Resumo por equipamento...")
    path = generate_equipment_summary_plot(df, output_dir)
    if path:
        files_generated.append(path)

    # Resumo
    print("\n" + "=" * 60)
    print("ANÁLISES AVANÇADAS CONCLUÍDAS")
    print("=" * 60)
    print(f"\nGráficos gerados: {len(files_generated)}")
    for f in files_generated:
        print(f"  - {f}")

    return {
        "status": "success",
        "plots_generated": len(files_generated),
        "files": files_generated
    }


if __name__ == "__main__":
    main()
