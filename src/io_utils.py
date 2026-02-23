from pathlib import Path
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
	return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str) -> None:
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(path, index=False)
	print(f"Guardado ({df.shape[0]} filas × {df.shape[1]} cols) en {path}")