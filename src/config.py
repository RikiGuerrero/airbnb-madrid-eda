from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Change these paths to point to your data files
RAW_PATH = ROOT / "data" / "raw" / "listings.csv"
OUT_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"

BASE_KEEP_COLS = {
	"id",
	"price",
	"room_type",
	"property_type",
	"neighbourhood_cleansed",
	"host_is_superhost",
	"accommodates",
	"bedrooms",
	"beds",
	"amenities",
	"number_of_reviews",
	"availability_365",
	"review_scores_rating"
}

# Columnas opcionales que enriquecen el análisis si están disponibles
OPTIONAL_COLS = {
	"host_listings_count",
	"minimum_nights",
	"maximum_nights",
	"reviews_per_month",
	"calculated_host_listings_count",
	"instant_bookable",
	"host_response_rate",
	"host_acceptance_rate",
}


def get_keep_cols(columns):
	"""
	Devuelve las columnas a conservar intersectando BASE_KEEP_COLS
	con las columnas disponibles, más cualquier columna opcional presente.

	Args:
		columns: Columnas disponibles en el DataFrame (Index o lista)

	Returns:
		Lista ordenada de columnas a conservar
	"""
	available = set(columns)

	# Columnas base que existen en el dataset
	keep = BASE_KEEP_COLS & available

	# Añadir columnas opcionales si están disponibles
	optional_found = OPTIONAL_COLS & available
	keep = keep | optional_found

	missing_base = BASE_KEEP_COLS - available
	if missing_base:
		print(f"[config] Columnas base no encontradas: {sorted(missing_base)}")

	if optional_found:
		print(f"[config] Columnas opcionales añadidas: {sorted(optional_found)}")

	return sorted(keep)