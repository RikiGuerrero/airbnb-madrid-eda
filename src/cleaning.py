import pandas as pd
import numpy as np


# ── Esquema de tipos objetivo ──────────────────────────────────────
DTYPE_SCHEMA = {
	'id':                     'int64',
	'price':                  'float64',
	'accommodates':           'int64',
	'bedrooms':               'float64',
	'beds':                   'float64',
	'availability_365':       'int64',
	'number_of_reviews':      'int64',
	'review_scores_rating':   'float64',
	'host_is_superhost':      'bool',
	'neighbourhood_cleansed': 'category',
	'property_type':          'category',
	'room_type':              'category'
}

def clean_price(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	
	if 'price' in out.columns:
		out['price'] = (
			out['price']
			.astype(str)
			.str.replace('$', '', regex=False)
			.str.strip()
		)
		out['price'] = pd.to_numeric(out['price'], errors='coerce')

		out.loc[out['price'] <= 0, 'price'] = np.nan

		print(f"'price' → float64  |  nulos tras conversión: {out['price'].isna().sum()}")
	
	return out

def clean_amenities(df: pd.DataFrame) -> pd.DataFrame:
	"""Parsea la columna amenities JSON y la elimina (el count se crea en features)."""
	out = df.copy()
	
	if 'amenities' in out.columns:
		import json
		
		def parse_amenities(x):
			if pd.isna(x):
				return []
			try:
				return json.loads(x)
			except Exception:
				return []
		
		out['amenities_parsed'] = out['amenities'].apply(parse_amenities)
		out.drop(columns=['amenities'], inplace=True)
		print(f"'amenities' → parseada (lista), columna original eliminada")
	
	return out

def clean_neighbourhood(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	
	for col in ['neighbourhood_cleansed', 'neighbourhood_group_cleansed']:
		if col in out.columns:
			out[col] = (
				out[col]
				.astype(str)
				.str.strip()
				.str.title()
				.replace('Nan', np.nan)
				.astype('category')
			)
	
	print(f"Barrios normalizados -> category")
	return out

def clean_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	
	bool_cols = ['host_is_superhost']
	
	for col in bool_cols:
		if col in out.columns:
			out[col] =(
				out[col]
				.astype(str)
				.str.strip()
				.str.lower()
				.map({'t': True, 'f': False})
			)
			print(f"'{col}' → bool  |  nulos: {out[col].isna().sum()}")

	return out

def clean_categorical(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	
	cat_cols = ['property_type', 'room_type']
	
	for col in cat_cols:
		if col in out.columns:
			out[col] = out[col].astype(str).str.strip().astype('category')
			print(f"'{col}' → category  |  niveles: {out[col].nunique()}")
	
	return out

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
	initial_shape = df.shape[0]
	out = df.drop_duplicates()
	removed = initial_shape - out.shape[0]
	print(f"Duplicados eliminados: {removed}")
	
	return out

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	print("Imputación de nulos")

	if out['price'].isna().any():
		before = out['price'].isna().sum()
		median_price = out.groupby(['room_type', 'neighbourhood_cleansed'])['price'].transform('median')
		out['price'] = out['price'].fillna(median_price)
		out['price'] = out['price'].fillna(out['price'].median())
		after = out['price'].isna().sum()
		print(f"    price:                {before} → {after} nulos  (mediana por room_type × barrio)")

	if 'bedrooms' in out.columns and out['bedrooms'].isna().any():
		before = out['bedrooms'].isna().sum()
		median_br = out.groupby(['room_type', 'accommodates'])['bedrooms'].transform('median')
		out['bedrooms'] = out['bedrooms'].fillna(median_br)
		out['bedrooms'] = out['bedrooms'].fillna(out['bedrooms'].median())
		after = out['bedrooms'].isna().sum()
		print(f"    bedrooms:             {before} → {after} nulos  (mediana por room_type × accommodates)")

	if 'beds' in out.columns and out['beds'].isna().any():
		before = out['beds'].isna().sum()
		median_beds = out.groupby(['room_type', 'accommodates'])['beds'].transform('median')
		out['beds'] = out['beds'].fillna(median_beds)
		out['beds'] = out['beds'].fillna(out['beds'].median())
		after = out['beds'].isna().sum()
		print(f"    beds:                 {before} → {after} nulos  (mediana por room_type × accommodates)")

	if 'review_scores_rating' in out.columns:
		n_miss = out['review_scores_rating'].isna().sum()
		print(f"    review_scores_rating: {n_miss} nulos conservados")

	if 'host_is_superhost' in out.columns and out['host_is_superhost'].isna().any():
		before = out['host_is_superhost'].isna().sum()
		out['host_is_superhost'] = out['host_is_superhost'].fillna(False)
		print(f"    host_is_superhost:    {before} → 0 nulos  (imputado como False)")

	return out

def drop_critical_nulls(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	before = len(out)
	out = out.dropna(subset=['price'])
	dropped = before - len(out)
	if dropped > 0:
		print(f"{dropped} filas sin precio eliminadas tras imputación")
	return out	

def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()

	for col, dtype in DTYPE_SCHEMA.items():
		if col not in out.columns:
			continue
		try:
			if dtype == 'bool':
				out[col] = out[col].astype('bool')
			elif dtype == 'category':
				out[col] = out[col].astype('category')
			else:
				out[col] = pd.to_numeric(out[col], errors='coerce').astype(dtype)
		except (ValueError, TypeError):
			print(f"No se pudo convertir '{col}' a {dtype}")

	print(f"Tipos finales aplicados según DTYPE_SCHEMA")
	return out

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Pipeline completo de limpieza.
	"""
	print("\n" + "="*50)
	print("INICIANDO LIMPIEZA")
	print("="*50)
	
	df = remove_duplicates(df)
	df = clean_boolean_columns(df)
	df = clean_price(df)
	df = clean_amenities(df)
	df = clean_neighbourhood(df)
	df = clean_categorical(df)
	df = impute_missing(df)
	df = drop_critical_nulls(df)
	df = enforce_dtypes(df)
	
	print(f"\n{'─' * 60}")
	print(f"LIMPIEZA COMPLETADA")
	print(f"  Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
	print(f"  Nulos restantes:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}" 
		  if df.isnull().sum().sum() > 0 else "  Nulos restantes: 0")
	print(f"{'─' * 60}")
	print(f"\nTipos finales:")
	print(df.dtypes.to_string())

	return df