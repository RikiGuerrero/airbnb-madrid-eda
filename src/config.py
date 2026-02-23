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


def get_keep_cols(columns):
	return sorted(BASE_KEEP_COLS)