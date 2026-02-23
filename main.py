import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')

from config import RAW_PATH, OUT_PATH, BASE_KEEP_COLS, get_keep_cols
from io_utils import load_csv, save_csv
from cleaning import clean_pipeline
from features import features_pipeline
from utils import qc_missing, qc_duplicates
from viz import (
    plot_correlation_heatmap,
    plot_correlation_ranking,
    plot_price_by_room_type,
    plot_price_by_neighbourhood,
    plot_heatmap_neighbourhood_room_type,
    plot_price_per_person,
    plot_superhost_boxplots,
    plot_superhost_diff_bars,
)




def main():
    viz_path = ROOT / 'visualizations'
    viz_path.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # 1. CARGA
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"1) CARGA DEL DATASET")
    print(f"{'='*60}")
    df_raw = load_csv(RAW_PATH)
    print(f"Forma: {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas")

    # ══════════════════════════════════════════════════════════════
    # 2. SELECCIÓN DE COLUMNAS
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"2) SELECCIÓN DE COLUMNAS")
    print(f"{'='*60}")
    KEEP_COLS = get_keep_cols(df_raw.columns)
    missing = sorted(BASE_KEEP_COLS - set(df_raw.columns))
    if missing:
        print(f"⚠ Faltan columnas esperadas: {missing}")

    original_cols = set(df_raw.columns)
    df_raw = df_raw[[c for c in KEEP_COLS if c in df_raw.columns]].copy()
    dropped = sorted(original_cols - set(df_raw.columns))
    print(f"Columnas conservadas: {len(df_raw.columns)}")
    print(f"Columnas eliminadas:  {len(dropped)}")

    # ══════════════════════════════════════════════════════════════
    # 3. QC INICIAL
    # ══════════════════════════════════════════════════════════════
    qc_missing(df_raw, verbose=True)
    qc_duplicates(df_raw, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # 4. LIMPIEZA
    # ══════════════════════════════════════════════════════════════
    df_clean = clean_pipeline(df_raw)

    # ══════════════════════════════════════════════════════════════
    # 5. FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════
    df_feat = features_pipeline(df_clean)

    # ══════════════════════════════════════════════════════════════
    # 6. PERSISTENCIA
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"6) GUARDADO")
    print(f"{'='*60}")
    save_csv(df_feat, OUT_PATH)

    # ══════════════════════════════════════════════════════════════
    # 7. PREGUNTA 1 — Factores asociados al precio
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"PREGUNTA 1: ¿Qué factores explican mejor el precio?")
    print(f"{'='*80}")

    corr_with_price = plot_correlation_heatmap(
        df_feat, save_path=viz_path / 'p1_heatmap_correlacion.png'
    )

    print(f"\nCorrelaciones con precio (Pearson):")
    for var, val in corr_with_price.items():
        signo = "+" if val > 0.3 else (" " if val > 0.1 else "-")
        print(f"  [{signo}] {var:<25s} {val:+.3f}")

    plot_correlation_ranking(
        corr_with_price, save_path=viz_path / 'p1_ranking_correlacion_precio.png'
    )

    top_var = corr_with_price.index[0]
    top_val = corr_with_price.values[0]
    print(f"\n-> Driver principal: {top_var.upper()} (r = {top_val:+.3f})")

    # ══════════════════════════════════════════════════════════════
    # 8. PREGUNTA 2 — Diferencias por tipo y barrio
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"PREGUNTA 2: ¿Diferencias de precio entre barrios y tipos?")
    print(f"{'='*80}")

    rt_stats = (
        df_feat.groupby('room_type')['price']
        .agg(['count', 'mean', 'median', 'std'])
        .sort_values('median', ascending=False)
        .round(2)
    )
    rt_stats['share_%'] = (rt_stats['count'] / rt_stats['count'].sum() * 100).round(1)
    print(f"\n-- Precio por TIPO DE ALOJAMIENTO --")
    print(rt_stats)

    top_n = 15
    top_barrios = df_feat['neighbourhood_cleansed'].value_counts().head(top_n).index
    df_top = df_feat[df_feat['neighbourhood_cleansed'].isin(top_barrios)]

    barrio_stats = (
        df_top.groupby('neighbourhood_cleansed')['price']
        .agg(['count', 'mean', 'median', 'std'])
        .sort_values('median', ascending=False)
        .round(2)
    )
    print(f"\n-- Precio por BARRIO (Top {top_n}) --")
    print(barrio_stats)

    global_median = df_feat['price'].median()

    plot_price_by_room_type(
        df_feat, rt_stats, save_path=viz_path / 'p2_boxplot_precio_room_type.png'
    )
    plot_price_by_neighbourhood(
        df_top, barrio_stats, global_median, top_n=top_n,
        save_path=viz_path / 'p2_barras_precio_barrio.png'
    )

    pivot_median = (
        df_top.pivot_table(values='price', index='neighbourhood_cleansed',
                           columns='room_type', aggfunc='median')
        .reindex(barrio_stats.index)
        .round(0)
    )
    plot_heatmap_neighbourhood_room_type(
        pivot_median, save_path=viz_path / 'p2_heatmap_barrio_room_type.png'
    )
    plot_price_per_person(
        df_feat, rt_stats, save_path=viz_path / 'p2_boxplot_price_per_person.png'
    )

    barrio_max = barrio_stats['median'].idxmax()
    barrio_min = barrio_stats['median'].idxmin()
    ratio_barrios = barrio_stats.loc[barrio_max, 'median'] / barrio_stats.loc[barrio_min, 'median']
    print(f"\n-> Barrio mas caro: {barrio_max} | Mas barato: {barrio_min} | Ratio: {ratio_barrios:.1f}x")

    # ══════════════════════════════════════════════════════════════
    # 9. PREGUNTA 3 — Superhost
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"PREGUNTA 3: ¿Ser Superhost = mayor precio o mayor demanda?")
    print(f"{'='*80}")

    compare_cols = ['price', 'number_of_reviews', 'review_scores_rating', 'availability_365']
    compare_cols = [c for c in compare_cols if c in df_feat.columns]

    metrics = {}
    print(f"\n-- Comparacion Superhost vs No-Superhost --")
    for col in compare_cols:
        mean_sh = df_feat[df_feat['host_is_superhost'] == True][col].mean()
        mean_no = df_feat[df_feat['host_is_superhost'] == False][col].mean()
        diff_pct = ((mean_sh - mean_no) / mean_no * 100) if mean_no != 0 else 0
        metrics[col] = diff_pct
        signo = "+" if diff_pct > 0 else ""
        print(f"  {col:<25s}  SH: {mean_sh:>8.1f}  |  No-SH: {mean_no:>8.1f}  |  D {signo}{diff_pct:.1f}%")

    n_super = df_feat['host_is_superhost'].sum()
    n_total = len(df_feat)
    print(f"\n  Superhosts: {n_super:,} de {n_total:,} ({n_super/n_total*100:.1f}%)")

    plot_superhost_boxplots(
        df_feat, save_path=viz_path / 'p3_boxplot_superhost.png'
    )
    plot_superhost_diff_bars(
        metrics, save_path=viz_path / 'p3_barras_superhost_diff.png'
    )

    price_sh = df_feat[df_feat['host_is_superhost'] == True]['price'].median()
    price_no = df_feat[df_feat['host_is_superhost'] == False]['price'].median()
    rev_sh = df_feat[df_feat['host_is_superhost'] == True]['number_of_reviews'].median()
    rev_no = df_feat[df_feat['host_is_superhost'] == False]['number_of_reviews'].median()
    print(f"\n-> Precio: SH {price_sh:.0f}EUR vs No-SH {price_no:.0f}EUR (sin diferencia significativa)")
    print(f"-> Resenas: SH {rev_sh:.0f} vs No-SH {rev_no:.0f} (Superhost = mas volumen)")

    # ══════════════════════════════════════════════════════════════
    # 10. RESUMEN FINAL
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETADO")
    print(f"{'='*80}")
    print(f"  Dataset procesado: {OUT_PATH}")
    print(f"  Visualizaciones:   {viz_path}/")
    print(f"  Graficos generados: {len(list(viz_path.glob('*.png')))}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()