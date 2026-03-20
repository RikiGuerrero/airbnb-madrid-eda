"""
Módulo de análisis con tablas resumen y pivots.
Proporciona funciones que generan tablas analíticas intermedias para respaldar las visualizaciones.
"""
import pandas as pd
import numpy as np


def summary_by_room_type(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Tabla resumen de métricas por tipo de alojamiento.

	Args:
		df: DataFrame con columnas room_type, price, number_of_reviews, etc.

	Returns:
		DataFrame con estadísticas agregadas por room_type
	"""
	agg_cols = {
		'price': ['count', 'mean', 'median', 'std', 'min', 'max'],
		'number_of_reviews': ['mean', 'median'],
		'review_scores_rating': ['mean'],
		'availability_365': ['mean']
	}

	# Filtrar solo columnas que existen
	agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}

	summary = df.groupby('room_type').agg(agg_cols).round(2)
	summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

	# Añadir porcentaje del total
	if 'price_count' in summary.columns:
		summary['share_pct'] = (summary['price_count'] / summary['price_count'].sum() * 100).round(1)

	return summary.sort_values('price_median', ascending=False)


def summary_by_neighbourhood(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
	"""
	Tabla resumen de métricas por barrio (top N por volumen).

	Args:
		df: DataFrame con columna neighbourhood_cleansed
		top_n: Número de barrios a incluir

	Returns:
		DataFrame con estadísticas agregadas por barrio
	"""
	top_barrios = df['neighbourhood_cleansed'].value_counts().head(top_n).index

	df_top = df[df['neighbourhood_cleansed'].isin(top_barrios)]

	agg_cols = {
		'price': ['count', 'mean', 'median', 'std'],
		'number_of_reviews': ['sum', 'mean'],
		'review_scores_rating': ['mean'],
		'host_is_superhost': ['mean']  # % de superhosts
	}

	agg_cols = {k: v for k, v in agg_cols.items() if k in df_top.columns}

	summary = df_top.groupby('neighbourhood_cleansed').agg(agg_cols).round(2)
	summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

	if 'host_is_superhost_mean' in summary.columns:
		summary = summary.rename(columns={'host_is_superhost_mean': 'superhost_pct'})
		summary['superhost_pct'] = (summary['superhost_pct'] * 100).round(1)

	return summary.sort_values('price_median', ascending=False)


def summary_by_price_bucket(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Tabla resumen de características por segmento de precio.

	Args:
		df: DataFrame con columna price_bucket

	Returns:
		DataFrame con perfil de cada segmento de precio
	"""
	if 'price_bucket' not in df.columns:
		return pd.DataFrame()

	agg_cols = {
		'price': ['count', 'mean', 'median'],
		'accommodates': ['mean'],
		'bedrooms': ['mean'],
		'amenities_count': ['mean'],
		'number_of_reviews': ['mean'],
		'review_scores_rating': ['mean'],
		'occupancy_ratio': ['mean']
	}

	agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}

	summary = df.groupby('price_bucket').agg(agg_cols).round(2)
	summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

	# Reordenar categorías
	order = ['budget', 'standard', 'premium', 'luxury']
	order = [o for o in order if o in summary.index]
	summary = summary.reindex(order)

	return summary


def summary_superhost_comparison(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Tabla comparativa Superhost vs No-Superhost.

	Args:
		df: DataFrame con columna host_is_superhost

	Returns:
		DataFrame con comparación lado a lado
	"""
	if 'host_is_superhost' not in df.columns:
		return pd.DataFrame()

	metrics = ['price', 'number_of_reviews', 'review_scores_rating',
			   'availability_365', 'amenities_count', 'occupancy_ratio']
	metrics = [m for m in metrics if m in df.columns]

	results = []
	for metric in metrics:
		sh_mean = df[df['host_is_superhost']][metric].mean()
		no_mean = df[~df['host_is_superhost']][metric].mean()
		sh_median = df[df['host_is_superhost']][metric].median()
		no_median = df[~df['host_is_superhost']][metric].median()

		diff_pct = ((sh_mean - no_mean) / no_mean * 100) if no_mean != 0 else 0

		results.append({
			'metric': metric,
			'superhost_mean': round(sh_mean, 2),
			'no_superhost_mean': round(no_mean, 2),
			'superhost_median': round(sh_median, 2),
			'no_superhost_median': round(no_median, 2),
			'diff_pct': round(diff_pct, 1)
		})

	summary = pd.DataFrame(results).set_index('metric')

	# Añadir conteos
	n_sh = df['host_is_superhost'].sum()
	n_total = len(df)
	summary.attrs['superhost_count'] = n_sh
	summary.attrs['superhost_pct'] = round(n_sh / n_total * 100, 1)

	return summary


def pivot_price_by_neighbourhood_roomtype(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
	"""
	Pivot table: precio mediano por barrio × tipo de alojamiento.

	Args:
		df: DataFrame
		top_n: Número de barrios a incluir

	Returns:
		DataFrame pivot con precio mediano
	"""
	top_barrios = df['neighbourhood_cleansed'].value_counts().head(top_n).index
	df_top = df[df['neighbourhood_cleansed'].isin(top_barrios)]

	pivot = pd.pivot_table(
		df_top,
		values='price',
		index='neighbourhood_cleansed',
		columns='room_type',
		aggfunc='median'
	).round(0)

	# Ordenar por precio mediano global
	medians = df_top.groupby('neighbourhood_cleansed')['price'].median()
	pivot = pivot.reindex(medians.sort_values(ascending=False).index)

	return pivot


def pivot_occupancy_by_neighbourhood_roomtype(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
	"""
	Pivot table: ocupación media por barrio × tipo de alojamiento.

	Args:
		df: DataFrame con occupancy_ratio
		top_n: Número de barrios a incluir

	Returns:
		DataFrame pivot con ratio de ocupación
	"""
	if 'occupancy_ratio' not in df.columns:
		return pd.DataFrame()

	top_barrios = df['neighbourhood_cleansed'].value_counts().head(top_n).index
	df_top = df[df['neighbourhood_cleansed'].isin(top_barrios)]

	pivot = pd.pivot_table(
		df_top,
		values='occupancy_ratio',
		index='neighbourhood_cleansed',
		columns='room_type',
		aggfunc='mean'
	).round(3)

	# Ordenar por ocupación media global
	means = df_top.groupby('neighbourhood_cleansed')['occupancy_ratio'].mean()
	pivot = pivot.reindex(means.sort_values(ascending=False).index)

	return pivot


def summary_by_host_type(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Tabla resumen por tipo de anfitrión (individual/small_host/professional).

	Args:
		df: DataFrame con columna host_type

	Returns:
		DataFrame con perfil de cada tipo de anfitrión
	"""
	if 'host_type' not in df.columns:
		return pd.DataFrame()

	agg_cols = {
		'price': ['count', 'mean', 'median'],
		'number_of_reviews': ['mean', 'sum'],
		'review_scores_rating': ['mean'],
		'availability_365': ['mean'],
		'occupancy_ratio': ['mean']
	}

	agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}

	summary = df.groupby('host_type').agg(agg_cols).round(2)
	summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

	# Reordenar
	order = ['individual', 'small_host', 'professional']
	order = [o for o in order if o in summary.index]
	summary = summary.reindex(order)

	return summary


def summary_correlations(df: pd.DataFrame, target: str = 'price') -> pd.DataFrame:
	"""
	Tabla de correlaciones con variable objetivo.

	Args:
		df: DataFrame
		target: Variable objetivo (default: 'price')

	Returns:
		DataFrame con correlaciones ordenadas
	"""
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

	if target not in numeric_cols:
		return pd.DataFrame()

	correlations = df[numeric_cols].corr()[target].drop(target)

	summary = pd.DataFrame({
		'correlation': correlations,
		'abs_correlation': correlations.abs(),
		'direction': correlations.apply(lambda x: 'positive' if x > 0 else 'negative')
	}).sort_values('abs_correlation', ascending=False).round(3)

	return summary


def run_analytics_suite(df: pd.DataFrame, verbose: bool = True) -> dict:
	"""
	Ejecuta suite completa de análisis y genera todas las tablas resumen.

	Args:
		df: DataFrame procesado
		verbose: Si True, imprime las tablas

	Returns:
		dict con todas las tablas generadas
	"""
	tables = {
		'by_room_type': summary_by_room_type(df),
		'by_neighbourhood': summary_by_neighbourhood(df),
		'by_price_bucket': summary_by_price_bucket(df),
		'superhost_comparison': summary_superhost_comparison(df),
		'pivot_price': pivot_price_by_neighbourhood_roomtype(df),
		'pivot_occupancy': pivot_occupancy_by_neighbourhood_roomtype(df),
		'by_host_type': summary_by_host_type(df),
		'correlations': summary_correlations(df)
	}

	if verbose:
		print("\n" + "=" * 70)
		print("TABLAS ANALÍTICAS")
		print("=" * 70)

		print("\n[1] RESUMEN POR TIPO DE ALOJAMIENTO")
		print("-" * 50)
		if not tables['by_room_type'].empty:
			print(tables['by_room_type'].to_string())

		print("\n[2] RESUMEN POR BARRIO (Top 15)")
		print("-" * 50)
		if not tables['by_neighbourhood'].empty:
			print(tables['by_neighbourhood'].to_string())

		print("\n[3] PERFIL POR SEGMENTO DE PRECIO")
		print("-" * 50)
		if not tables['by_price_bucket'].empty:
			print(tables['by_price_bucket'].to_string())

		print("\n[4] COMPARATIVA SUPERHOST")
		print("-" * 50)
		if not tables['superhost_comparison'].empty:
			print(tables['superhost_comparison'].to_string())
			if hasattr(tables['superhost_comparison'], 'attrs'):
				print(f"\nSuperhosts: {tables['superhost_comparison'].attrs.get('superhost_count', 'N/A')} "
					  f"({tables['superhost_comparison'].attrs.get('superhost_pct', 'N/A')}%)")

		print("\n[5] PIVOT: PRECIO MEDIANO (Barrio × Tipo)")
		print("-" * 50)
		if not tables['pivot_price'].empty:
			print(tables['pivot_price'].to_string())

		print("\n[6] PERFIL POR TIPO DE ANFITRIÓN")
		print("-" * 50)
		if not tables['by_host_type'].empty:
			print(tables['by_host_type'].to_string())

		print("\n[7] CORRELACIONES CON PRECIO")
		print("-" * 50)
		if not tables['correlations'].empty:
			print(tables['correlations'].to_string())

		print("\n" + "=" * 70)

	return tables
