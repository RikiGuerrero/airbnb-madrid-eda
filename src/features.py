import pandas as pd
import numpy as np


def create_amenities_feature(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Crea amenities_count a partir de la columna parseada en cleaning.
	"""
	out = df.copy()

	if 'amenities_parsed' in out.columns:
		out['amenities_count'] = out['amenities_parsed'].apply(len).astype('int64')
		out.drop(columns=['amenities_parsed'], inplace=True)
		print(f"  amenities_count creada (int64)")
	else:
		print("  amenities_parsed no encontrada, se omite amenities_count")

	return out


def create_capacity_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Crea features de capacidad:
	- price_per_person: price / accommodates
	"""
	out = df.copy()

	if {'price', 'accommodates'}.issubset(out.columns):
		out['price_per_person'] = (
			out['price'] / out['accommodates'].replace(0, 1)
		).round(2)
		print(f"  price_per_person creada")

	return out


def create_price_bucket(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Crea price_bucket categorizando el precio en segmentos:
	- budget: percentil 0-25
	- standard: percentil 25-50
	- premium: percentil 50-75
	- luxury: percentil 75-100
	"""
	out = df.copy()

	if 'price' in out.columns:
		q25, q50, q75 = out['price'].quantile([0.25, 0.50, 0.75])

		def categorize_price(p):
			if pd.isna(p):
				return np.nan
			elif p <= q25:
				return 'budget'
			elif p <= q50:
				return 'standard'
			elif p <= q75:
				return 'premium'
			else:
				return 'luxury'

		out['price_bucket'] = out['price'].apply(categorize_price).astype('category')
		print(f"  price_bucket creada (budget/standard/premium/luxury)")
		print(f"    Cortes: budget<={q25:.0f}€, standard<={q50:.0f}€, premium<={q75:.0f}€, luxury>{q75:.0f}€")

	return out


def create_occupancy_ratio(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Crea occupancy_ratio estimando la ocupación a partir de availability_365:
	- occupancy_ratio = (365 - availability_365) / 365
	Valores altos indican mayor demanda/ocupación.
	"""
	out = df.copy()

	if 'availability_365' in out.columns:
		out['occupancy_ratio'] = (
			(365 - out['availability_365'].clip(0, 365)) / 365
		).round(3)
		print(f"  occupancy_ratio creada (0-1, mayor = más ocupado)")

	return out


def create_rating_bucket(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Crea rating_bucket categorizando las puntuaciones:
	- sin_rating: NaN
	- bajo: < 4.0
	- medio: 4.0 - 4.5
	- alto: 4.5 - 4.8
	- excelente: >= 4.8
	"""
	out = df.copy()

	if 'review_scores_rating' in out.columns:
		def categorize_rating(r):
			if pd.isna(r):
				return 'sin_rating'
			elif r < 4.0:
				return 'bajo'
			elif r < 4.5:
				return 'medio'
			elif r < 4.8:
				return 'alto'
			else:
				return 'excelente'

		out['rating_bucket'] = out['review_scores_rating'].apply(categorize_rating).astype('category')

		dist = out['rating_bucket'].value_counts()
		print(f"  rating_bucket creada (sin_rating/bajo/medio/alto/excelente)")
		print(f"    Distribución: {dict(dist)}")

	return out


def create_host_type(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Crea host_type clasificando al anfitrión según número de listings:
	- individual: 1 listing
	- small_host: 2-5 listings
	- professional: 6+ listings

	Requiere columna host_listings_count o calculated_host_listings_count.
	"""
	out = df.copy()

	host_col = None
	for col in ['host_listings_count', 'calculated_host_listings_count']:
		if col in out.columns:
			host_col = col
			break

	if host_col:
		def categorize_host(n):
			if pd.isna(n) or n < 1:
				return 'individual'
			elif n == 1:
				return 'individual'
			elif n <= 5:
				return 'small_host'
			else:
				return 'professional'

		out['host_type'] = out[host_col].apply(categorize_host).astype('category')

		dist = out['host_type'].value_counts()
		print(f"  host_type creada (basada en {host_col})")
		print(f"    Distribución: {dict(dist)}")

	return out


def features_pipeline(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Pipeline completo de feature engineering.
	"""
	print("\n" + "="*50)
	print("INICIANDO FEATURE ENGINEERING")
	print("="*50 + "\n")

	df = create_amenities_feature(df)
	df = create_capacity_features(df)
	df = create_price_bucket(df)
	df = create_occupancy_ratio(df)
	df = create_rating_bucket(df)
	df = create_host_type(df)

	n_features_new = sum(1 for c in ['amenities_count', 'price_per_person', 'price_bucket',
									  'occupancy_ratio', 'rating_bucket', 'host_type'] if c in df.columns)
	print(f"\nFeature engineering completado ({n_features_new} features creadas)")
	return df