import pandas as pd

def qc_missing(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
	missing = df.isnull().sum()
	missing_pct = (missing / len(df) * 100).round(2)
	
	result = pd.DataFrame({
		'Faltantes': missing,
		'Porcentaje (%)': missing_pct
	}).sort_values('Faltantes', ascending=False)
	
	result = result[result['Faltantes'] > 0]
	if verbose and len(result) > 0:
		print("\n" + "="*50)
		print("VALORES FALTANTES")
		print("="*50)
		print(result)
		
	return result

def qc_duplicates(df: pd.DataFrame, verbose: bool = True) -> int:
	dup_count = df.duplicated().sum()
	
	if verbose:
		print(f"\nDuplicados detectados: {dup_count}")
		
	return dup_count