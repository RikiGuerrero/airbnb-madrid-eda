"""
Módulo de validación post-limpieza.
Proporciona checks explícitos para verificar la calidad del dataset tras el procesamiento.
"""
import pandas as pd
import numpy as np


def validate_nulls(df: pd.DataFrame, critical_cols: list = None) -> dict:
	"""
	Valida nulos en el dataset y reporta el estado final.

	Args:
		df: DataFrame a validar
		critical_cols: Columnas que NO deberían tener nulos

	Returns:
		dict con el reporte de validación
	"""
	if critical_cols is None:
		critical_cols = ['price', 'id', 'room_type']

	null_counts = df.isnull().sum()
	null_pct = (null_counts / len(df) * 100).round(2)

	report = {
		'total_nulls': null_counts.sum(),
		'cols_with_nulls': null_counts[null_counts > 0].to_dict(),
		'null_percentages': null_pct[null_pct > 0].to_dict(),
		'critical_violations': [],
		'status': 'OK'
	}

	for col in critical_cols:
		if col in df.columns and null_counts.get(col, 0) > 0:
			report['critical_violations'].append(col)
			report['status'] = 'WARNING'

	return report


def validate_duplicates(df: pd.DataFrame, id_col: str = 'id') -> dict:
	"""
	Valida duplicados en el dataset.

	Args:
		df: DataFrame a validar
		id_col: Columna de identificador único

	Returns:
		dict con el reporte de validación
	"""
	# Excluir columnas con tipos no hashables (listas, dicts)
	hashable_cols = []
	for col in df.columns:
		try:
			# Verificar si la columna contiene tipos hashables
			sample = df[col].dropna().head(1)
			if len(sample) > 0:
				hash(sample.iloc[0])
			hashable_cols.append(col)
		except TypeError:
			pass

	df_hashable = df[hashable_cols] if hashable_cols else df[[id_col]]

	total_dups = df_hashable.duplicated().sum()
	id_dups = df.duplicated(subset=[id_col]).sum() if id_col in df.columns else 0

	report = {
		'total_duplicates': total_dups,
		'id_duplicates': id_dups,
		'status': 'OK' if total_dups == 0 and id_dups == 0 else 'WARNING'
	}

	return report


def validate_distributions(df: pd.DataFrame, df_original: pd.DataFrame = None) -> dict:
	"""
	Valida distribuciones de variables críticas.
	Si se proporciona df_original, compara antes/después.

	Args:
		df: DataFrame procesado
		df_original: DataFrame original (opcional)

	Returns:
		dict con estadísticas de distribución
	"""
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	critical_numeric = [c for c in ['price', 'bedrooms', 'beds', 'accommodates'] if c in numeric_cols]

	report = {'distributions': {}, 'comparisons': {}}

	for col in critical_numeric:
		stats = {
			'mean': df[col].mean(),
			'median': df[col].median(),
			'std': df[col].std(),
			'min': df[col].min(),
			'max': df[col].max(),
			'q25': df[col].quantile(0.25),
			'q75': df[col].quantile(0.75)
		}
		report['distributions'][col] = {k: round(v, 2) for k, v in stats.items()}

		if df_original is not None and col in df_original.columns:
			original_median = pd.to_numeric(df_original[col], errors='coerce').median()
			if not pd.isna(original_median) and original_median != 0:
				change_pct = ((stats['median'] - original_median) / original_median * 100)
				report['comparisons'][col] = {
					'original_median': round(original_median, 2),
					'final_median': round(stats['median'], 2),
					'change_pct': round(change_pct, 2)
				}

	return report


def validate_dtypes(df: pd.DataFrame, expected_schema: dict = None) -> dict:
	"""
	Valida que los tipos de datos sean los esperados.

	Args:
		df: DataFrame a validar
		expected_schema: diccionario {columna: tipo_esperado}

	Returns:
		dict con el reporte de validación
	"""
	if expected_schema is None:
		expected_schema = {
			'price': 'float64',
			'accommodates': 'int64',
			'host_is_superhost': 'bool'
		}

	report = {'matches': [], 'mismatches': [], 'status': 'OK'}

	for col, expected in expected_schema.items():
		if col not in df.columns:
			continue

		actual = str(df[col].dtype)
		if expected in actual or actual in expected:
			report['matches'].append(col)
		else:
			report['mismatches'].append({'column': col, 'expected': expected, 'actual': actual})
			report['status'] = 'WARNING'

	return report


def validate_cardinality(df: pd.DataFrame) -> dict:
	"""
	Valida cardinalidad de variables categóricas.

	Args:
		df: DataFrame a validar

	Returns:
		dict con información de cardinalidad
	"""
	cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()

	report = {}
	for col in cat_cols:
		try:
			# Verificar si la columna contiene tipos hashables
			sample = df[col].dropna().head(1)
			if len(sample) > 0:
				hash(sample.iloc[0])

			nunique = df[col].nunique()
			top_values = df[col].value_counts().head(5).to_dict()
			report[col] = {
				'unique_values': nunique,
				'top_5': top_values
			}
		except TypeError:
			# Columna contiene tipos no hashables (listas, dicts)
			report[col] = {
				'unique_values': 'N/A (unhashable)',
				'top_5': {}
			}

	return report


def run_validation_suite(df: pd.DataFrame, df_original: pd.DataFrame = None,
						  verbose: bool = True) -> dict:
	"""
	Ejecuta suite completa de validación post-limpieza.

	Args:
		df: DataFrame procesado
		df_original: DataFrame original (opcional, para comparaciones)
		verbose: Si True, imprime el reporte

	Returns:
		dict con todos los reportes de validación
	"""
	report = {
		'nulls': validate_nulls(df),
		'duplicates': validate_duplicates(df),
		'distributions': validate_distributions(df, df_original),
		'dtypes': validate_dtypes(df),
		'cardinality': validate_cardinality(df)
	}

	# Estado general
	all_ok = all(
		r.get('status', 'OK') == 'OK'
		for r in [report['nulls'], report['duplicates'], report['dtypes']]
	)
	report['overall_status'] = 'PASS' if all_ok else 'REVIEW'

	if verbose:
		print("\n" + "=" * 60)
		print("VALIDACIÓN POST-LIMPIEZA")
		print("=" * 60)

		# Nulos
		print(f"\n[Nulos]")
		print(f"  Total nulos: {report['nulls']['total_nulls']}")
		if report['nulls']['cols_with_nulls']:
			for col, count in report['nulls']['cols_with_nulls'].items():
				pct = report['nulls']['null_percentages'].get(col, 0)
				print(f"    {col}: {count} ({pct}%)")
		if report['nulls']['critical_violations']:
			print(f"  ⚠ Violaciones críticas: {report['nulls']['critical_violations']}")

		# Duplicados
		print(f"\n[Duplicados]")
		print(f"  Filas duplicadas: {report['duplicates']['total_duplicates']}")
		print(f"  IDs duplicados: {report['duplicates']['id_duplicates']}")

		# Distribuciones
		print(f"\n[Distribuciones variables críticas]")
		for col, stats in report['distributions']['distributions'].items():
			print(f"  {col}: media={stats['mean']}, mediana={stats['median']}, "
				  f"rango=[{stats['min']}, {stats['max']}]")

		if report['distributions']['comparisons']:
			print(f"\n[Comparación antes/después]")
			for col, comp in report['distributions']['comparisons'].items():
				print(f"  {col}: {comp['original_median']} → {comp['final_median']} "
					  f"({comp['change_pct']:+.1f}%)")

		# Cardinalidad
		print(f"\n[Cardinalidad categóricas]")
		for col, info in report['cardinality'].items():
			print(f"  {col}: {info['unique_values']} valores únicos")

		# Estado final
		status_icon = "✓" if report['overall_status'] == 'PASS' else "⚠"
		print(f"\n{status_icon} Estado general: {report['overall_status']}")
		print("=" * 60)

	return report
