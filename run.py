from gcj_loader import load_gcj_dataset
from main import create_dataframe
from classifier import run_full_experiment_suite

dataset = load_gcj_dataset(
    csv_paths=["gcj2009.csv"],
    language="py",
    min_snippets=8,
    max_authors=50,
    max_snippets_per_author=12
)

if dataset:
    df = create_dataframe(dataset)
    run_full_experiment_suite(df)