import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# PREGUNTA 1 — Factores asociados al precio
# ══════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path = None) -> pd.Series:
	"""
	Heatmap triangular de correlación entre variables numéricas.
	Retorna la serie de correlaciones con precio (sin precio).
	"""
	numeric_cols = [
		'price', 'accommodates', 'bedrooms', 'beds',
		'number_of_reviews', 'review_scores_rating',
		'availability_365', 'amenities_count'
	]
	numeric_cols = [c for c in numeric_cols if c in df.columns]

	corr_matrix = df[numeric_cols].corr()
	corr_with_price = corr_matrix['price'].drop('price').sort_values(ascending=False)

	fig, ax = plt.subplots(figsize=(10, 8))
	mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
	sns.heatmap(
		corr_matrix, mask=mask, annot=True, fmt='.2f',
		cmap='RdBu_r', center=0, vmin=-1, vmax=1,
		square=True, linewidths=0.5,
		cbar_kws={'label': 'Correlación de Pearson'},
		ax=ax
	)
	ax.set_title('Matriz de correlación — Variables numéricas',
				 fontsize=14, fontweight='bold', pad=15)
	fig.tight_layout()

	if save_path:
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"Guardado: {save_path.name}")

	plt.show()
	return corr_with_price


def plot_correlation_ranking(corr_with_price: pd.Series, save_path: Path = None) -> None:
	"""
	Barras horizontales con ranking de correlación con precio.
	"""
	fig, ax = plt.subplots(figsize=(10, 5))
	colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in corr_with_price.values]
	bars = ax.barh(corr_with_price.index, corr_with_price.values,
				   color=colors, edgecolor='white', height=0.6)

	for bar, val in zip(bars, corr_with_price.values):
		offset = 0.02 if val > 0 else -0.02
		ha = 'left' if val > 0 else 'right'
		ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
				f'{val:+.3f}', va='center', ha=ha, fontsize=10, fontweight='bold')

	ax.axvline(0, color='black', linewidth=0.8)
	ax.set_xlabel('Correlación de Pearson con precio', fontsize=11)
	ax.set_title('Ranking de correlación con precio',
				 fontsize=14, fontweight='bold', pad=15)
	ax.set_xlim(-0.5, 0.7)
	ax.invert_yaxis()
	sns.despine(left=True, bottom=True)
	fig.tight_layout()

	if save_path:
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"Guardado: {save_path.name}")

	plt.show()


# ══════════════════════════════════════════════════════════════════════
# PREGUNTA 2 — Diferencias de precio por tipo y barrio
# ══════════════════════════════════════════════════════════════════════

def plot_price_by_room_type(df: pd.DataFrame, rt_stats: pd.DataFrame,
							save_path: Path = None) -> None:
	"""
	Boxplot de precio por room_type con anotaciones de mediana y count.
	"""
	order_rt = rt_stats.index.tolist()
	palette_rt = {
		'Entire home/apt': '#e74c3c', 'Private room': '#3498db',
		'Shared room': '#2ecc71', 'Hotel room': '#f39c12'
	}

	fig, ax = plt.subplots(figsize=(12, 6))
	sns.boxplot(data=df, x='room_type', y='price', order=order_rt,
				hue='room_type', palette=palette_rt, showfliers=False,
				legend=False, ax=ax)

	for i, rt in enumerate(order_rt):
		med = rt_stats.loc[rt, 'median']
		cnt = int(rt_stats.loc[rt, 'count'])
		ax.text(i, med + 8, f'€{med:.0f}', ha='center',
				fontweight='bold', fontsize=11, color='black')
		ax.text(i, ax.get_ylim()[1] * 0.92, f'n={cnt:,}',
				ha='center', fontsize=9, color='gray')

	ax.set_title('Distribución de precio por Tipo de Alojamiento',
				 fontsize=14, fontweight='bold', pad=15)
	ax.set_xlabel('')
	ax.set_ylabel('Precio (€/noche)', fontsize=11)
	fig.tight_layout()

	if save_path:
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"Guardado: {save_path.name}")

	plt.show()


def plot_price_by_neighbourhood(df: pd.DataFrame, barrio_stats: pd.DataFrame,
								global_median: float, top_n: int = 15,
								save_path: Path = None) -> None:
	"""
	Barras horizontales de precio mediano por barrio (Top N).
	"""
	barrio_order = barrio_stats.sort_values('median', ascending=True)
	colors_barrio = plt.cm.YlOrRd(np.linspace(0.25, 0.85, len(barrio_order)))

	fig, ax = plt.subplots(figsize=(12, 8))
	bars = ax.barh(barrio_order.index, barrio_order['median'],
				   color=colors_barrio, edgecolor='white', height=0.7)

	for bar, (barrio, row) in zip(bars, barrio_order.iterrows()):
		val = row['median']
		cnt = int(row['count'])
		ax.text(val + 3, bar.get_y() + bar.get_height() / 2,
				f'€{val:.0f}  (n={cnt:,})', va='center', ha='left',
				fontsize=9, fontweight='bold')

	ax.axvline(global_median, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
	ax.text(global_median + 2, ax.get_ylim()[1] * 0.98,
			f'Mediana global: €{global_median:.0f}',
			fontsize=9, color='red', fontweight='bold', va='top')

	ax.set_xlabel('Precio mediano (€/noche)', fontsize=11)
	ax.set_title(f'Precio mediano por Barrio (Top {top_n} por volumen)',
				 fontsize=14, fontweight='bold', pad=15)
	sns.despine(left=True, bottom=True)
	fig.tight_layout()

	if save_path:
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"Guardado: {save_path.name}")

	plt.show()


def plot_heatmap_neighbourhood_room_type(pivot_median: pd.DataFrame,
										  save_path: Path = None) -> None:
	"""
	Heatmap cruzado barrio × room_type (precio mediano).
	"""
	fig, ax = plt.subplots(figsize=(12, 9))
	sns.heatmap(pivot_median, annot=True, fmt='.0f', cmap='YlOrRd',
				linewidths=0.5, cbar_kws={'label': 'Precio mediano (€)'}, ax=ax)
	ax.set_title('Precio mediano (€) — Barrio × Tipo de Alojamiento',
				 fontsize=14, fontweight='bold', pad=15)
	ax.set_xlabel('')
	ax.set_ylabel('')
	fig.tight_layout()

	if save_path:
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"Guardado: {save_path.name}")

	plt.show()


def plot_price_per_person(df: pd.DataFrame, rt_stats: pd.DataFrame,
						  save_path: Path = None) -> None:
	"""
	Boxplot de price_per_person por room_type.
	"""
	if 'price_per_person' not in df.columns:
		print("price_per_person no encontrada, se omite gráfico")
		return

	order_rt = rt_stats.index.tolist()
	palette_rt = {
		'Entire home/apt': '#e74c3c', 'Private room': '#3498db',
		'Shared room': '#2ecc71', 'Hotel room': '#f39c12'
	}

	fig, ax = plt.subplots(figsize=(12, 6))
	sns.boxplot(data=df, x='room_type', y='price_per_person', order=order_rt,
				hue='room_type', palette=palette_rt, showfliers=False,
				legend=False, ax=ax)

	for i, rt in enumerate(order_rt):
		med_pp = df[df['room_type'] == rt]['price_per_person'].median()
		ax.text(i, med_pp + 3, f'€{med_pp:.0f}/pers', ha='center',
				fontweight='bold', fontsize=11)

	ax.set_title('Precio por persona por Tipo de Alojamiento',
				 fontsize=14, fontweight='bold', pad=15)
	ax.set_xlabel('')
	ax.set_ylabel('Precio por persona (€/noche)', fontsize=11)
	fig.tight_layout()

	if save_path:
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"Guardado: {save_path.name}")

	plt.show()


# ══════════════════════════════════════════════════════════════════════
# PREGUNTA 3 — Superhost: precio vs demanda
# ══════════════════════════════════════════════════════════════════════

def plot_superhost_boxplots(df: pd.DataFrame, save_path: Path = None) -> None:
	"""
	Triple boxplot: precio, reseñas y rating — Superhost vs No-Superhost.
	"""
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	palette_sh = {True: '#2ecc71', False: '#e74c3c'}

	# ── Precio ──
	sns.boxplot(data=df, x='host_is_superhost', y='price',
				hue='host_is_superhost', palette=palette_sh,
				showfliers=False, legend=False, ax=axes[0])
	for i, val in enumerate([False, True]):
		med = df[df['host_is_superhost'] == val]['price'].median()
		axes[0].text(i, med + 5, f'€{med:.0f}', ha='center',
					 fontweight='bold', fontsize=11)
	axes[0].set_xticks([0, 1])
	axes[0].set_xticklabels(['No Superhost', 'Superhost'])
	axes[0].set_xlabel('')
	axes[0].set_ylabel('Precio (€/noche)')
	axes[0].set_title('Precio', fontsize=13, fontweight='bold')

	# ── Nº de reseñas ──
	sns.boxplot(data=df, x='host_is_superhost', y='number_of_reviews',
				hue='host_is_superhost', palette=palette_sh,
				showfliers=False, legend=False, ax=axes[1])
	for i, val in enumerate([False, True]):
		med = df[df['host_is_superhost'] == val]['number_of_reviews'].median()
		axes[1].text(i, med + 3, f'{med:.0f}', ha='center',
					 fontweight='bold', fontsize=11)
	axes[1].set_xticks([0, 1])
	axes[1].set_xticklabels(['No Superhost', 'Superhost'])
	axes[1].set_xlabel('')
	axes[1].set_ylabel('Nº de reseñas')
	axes[1].set_title('Volumen de reseñas (proxy demanda)', fontsize=13, fontweight='bold')

	# ── Rating ──
	sns.boxplot(data=df[df['review_scores_rating'].notna()],
				x='host_is_superhost', y='review_scores_rating',
				hue='host_is_superhost', palette=palette_sh,
				showfliers=False, legend=False, ax=axes[2])
	for i, val in enumerate([False, True]):
		med = df[df['host_is_superhost'] == val]['review_scores_rating'].median()
		if pd.notna(med):
			axes[2].text(i, med + 0.02, f'{med:.2f}', ha='center',
						 fontweight='bold', fontsize=11)
	axes[2].set_xticks([0, 1])
	axes[2].set_xticklabels(['No Superhost', 'Superhost'])
	axes[2].set_xlabel('')
	axes[2].set_ylabel('Rating')
	axes[2].set_title('Valoración media', fontsize=13, fontweight='bold')

	fig.suptitle('Superhost vs No-Superhost: Precio, Demanda y Calidad',
				 fontsize=15, fontweight='bold', y=1.02)
	fig.tight_layout()

	if save_path:
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"Guardado: {save_path.name}")

	plt.show()


def plot_superhost_diff_bars(metrics: dict, save_path: Path = None) -> None:
	"""
	Barras horizontales de diferencia porcentual Superhost vs No-Superhost.
	"""
	fig, ax = plt.subplots(figsize=(10, 5))
	names = list(metrics.keys())
	values = list(metrics.values())
	colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

	bars = ax.barh(names, values, color=colors, edgecolor='white', height=0.6)

	for bar, val in zip(bars, values):
		offset = 1 if val > 0 else -1
		ha = 'left' if val > 0 else 'right'
		ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
				f'{val:+.1f}%', va='center', ha=ha, fontsize=11, fontweight='bold')

	ax.axvline(0, color='black', linewidth=0.8)
	ax.set_xlabel('Diferencia porcentual (Superhost vs No-Superhost)', fontsize=11)
	ax.set_title('Impacto del estatus Superhost en métricas clave',
				 fontsize=14, fontweight='bold', pad=15)
	ax.invert_yaxis()
	sns.despine(left=True, bottom=True)
	fig.tight_layout()

	if save_path:
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"Guardado: {save_path.name}")

	plt.show()