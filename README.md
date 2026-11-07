# YC Startup Feature Extraction

An LLM pipeline that turns free-text company descriptions into a clean, categorical feature table for startup success modeling.

- **Success (label = 1):** 5,833 Y Combinator companies
- **Failure (label = 0):** 1,761 shut-down AI tools from the dang.ai *AI Graveyard*

## Pipeline

| Step | Notebook | What it does |
|---|---|---|
| 1 | `01_collect_and_extract_features.ipynb` | Pulls all YC companies from the public [yc-oss API](https://github.com/yc-oss/api) and scrapes the AI Graveyard (list + detail pages) into MySQL. It then builds one `unified_text` per company. |
| 2 | ″ | Embeds 7,730 texts with `BAAI/bge-m3` (1024-d) and runs K-Means with k = 80. The company nearest each centroid forms an 80-sample **golden set**. |
| 3 | ″ | **Schema discovery**: GPT-4o proposes 8–10 orthogonal business dimensions from the golden set. Synonym/overlap checks run in Python, and the result is hand-curated into [`data/schema_v1.json`](data/schema_v1.json). |
| 4 | ″ | **Feature extraction**: each dimension becomes a Pydantic `Enum`, so the LLM can only answer with allowed values (fallback `other_unknown`). Two runs were made: a GPT-4o-mini sandbox on 100 rows and a DeepSeek-chat run on the full set. Calls go out in async micro-batches with rate-limit sleeps, and each chunk is upserted into MySQL so the job resumes after interruption. |
| 5 | ″ | Labels rows: YC = 1, Graveyard = 0. Names that appear in both lists are treated as failures. |
| 6 | `02_build_lookup_tables.ipynb` | Normalizes the flat table into `*_lookup` tables plus `company_feature_keys` (integer foreign keys) for modeling. |
| 7 | `03_export_features.ipynb` | Exports the final table to CSV. |
| 8 | `04_success_classifier.ipynb` | Drops the 249 parse-error rows (7,345 left), screens each tag with chi-square tests, odds ratios and Wilson lower bounds, then trains Random Forest classifiers on binary- and one-hot-encoded features (5-fold `GridSearchCV`, balanced class weights, stratified 80/20 split) and explains the one-hot model with SHAP. |

## Success classifier (notebook 04)

Test set: 1,469 companies (stratified 20 %). Always predicting "success" would score 76.2 % accuracy.

| Encoding | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Binary (`category_encoders`) | 0.920 | 0.954 | 0.940 | 0.947 | 0.963 |
| **One-hot** | **0.922** | **0.956** | **0.941** | **0.949** | **0.965** |

By total absolute SHAP impact, `core_technology` ranks first, followed by `industry`, `product_type` and `target_audience`. Read these results together with the source bias described under *Known limitations*: part of the signal reflects which list a company came from.

## Extracted dimensions

| Dimension | # values | Most common | `other_unknown` |
|---|---|---|---|
| industry | 8 | other_unknown | 29.5 % |
| core_technology | 6 | ai_ml (55.5 %) | 32.5 % |
| product_type | 5 | software_platform (47.0 %) | 12.2 % |
| target_audience | 5 | enterprises (47.6 %) | 5.5 % |
| monetization_strategy | 7 | subscription (42.0 %) | 27.1 % |
| delivery_medium | 7 | web_application (59.7 %) | 12.5 % |
| value_proposition | 6 | workflow_automation (50.4 %) | 14.8 % |
| regulatory_environment | 5 | unregulated_or_low_friction (60.7 %) | 6.0 % |

7,594 of 7,730 companies are kept. `company_name` is the primary key, so names that differ only in case are collapsed. 249 rows fell back to `other_unknown` after a DeepSeek parse error; their `error` column is set.

## Data (`data/`)

| File | Rows | Description |
|---|---|---|
| `df_all_formatted.csv` | 7,730 | `company_name`, `unified_text` |
| `df_golden.csv` | 80 | Cluster-centroid samples used for schema discovery |
| `schema_v1.json` | – | Final 8-dimension schema with allowed values |
| `sandbox_eval_clean.csv` | 50 | DeepSeek sandbox output used to check quality before the full run |
| `final_yc_extracted_features.csv` | 7,594 | Final features + `label` |

`embeddings_cache.npy` (7,730 × 1024, ~31 MB) is not committed. Re-create it with the embedding cell in notebook 01.

## Known limitations

- **Source bias.** Every failure comes from an AI-tool directory, so `core_technology = ai_ml` covers 96.8 % of label 0 but only 43.0 % of label 1. Graveyard texts also contain phrases such as "Dang.ai no longer lists …". A model trained on this table would partly learn *which source a row came from*. A fair evaluation should control for this, for example by comparing within AI companies only or by stripping source-specific phrases.
- **Label noise.** YC companies marked *Inactive* are still labeled 1.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your own keys / MySQL credentials
```

Run the notebooks from the `notebooks/` folder; they read and write `../data/`. Notebook 04 reads `../data/final_yc_extracted_features.csv` and writes its intermediate CSVs and saved models into `notebooks/` (git-ignored).
