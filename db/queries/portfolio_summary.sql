WITH ranked AS (
    SELECT p.*, ROW_NUMBER() OVER (
        PARTITION BY model_id, dataset_id, provenance, application_key
        ORDER BY created_at DESC, id DESC
    ) AS occurrence
    FROM predictions p
    WHERE model_id = :model_id AND dataset_id = :dataset_id AND provenance = :provenance
), latest AS (
    SELECT *, COALESCE(calibrated_probability, raw_probability) AS probability
    FROM ranked WHERE occurrence = 1
)
SELECT risk_band, COUNT(*) AS applications, AVG(probability) AS mean_probability,
       COUNT(*) FILTER (WHERE calibrated_probability IS NOT NULL) AS calibrated_count
FROM latest GROUP BY risk_band ORDER BY risk_band
