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
		print(f"amenities_count creada (int64)")
	else:
		print("amenities_parsed no encontrada, se omite amenities_count")

	return out


def create_capacity_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Crea features de capacidad:
	- price_per_person: price / accommodates
	"""
	out = df.copy()

	if set(['price', 'accommodates']).issubset(out.columns):
		out['price_per_person'] = (
			out['price'] / out['accommodates'].replace(0, 1)
		).round(2)
		print(f"price_per_person creada")

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

	print("\nFeature engineering completado\n")
	return df