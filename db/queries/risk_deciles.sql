WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY application_key ORDER BY created_at DESC, id DESC) AS rn
    FROM predictions WHERE model_id = :model_id AND provenance = :provenance
), bands AS (
    SELECT NTILE(10) OVER (ORDER BY COALESCE(calibrated_probability, raw_probability), application_key) AS decile,
           COALESCE(calibrated_probability, raw_probability) AS probability
    FROM ranked WHERE rn = 1
)
SELECT decile, COUNT(*) AS applications, AVG(probability) AS mean_probability
FROM bands GROUP BY decile ORDER BY decile
