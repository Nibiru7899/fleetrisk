WITH source_rows AS (
    SELECT s.source_id, s.provenance, a.target,
           COALESCE(a.attributes->>'branch_id', a.attributes->>'NAME_CONTRACT_TYPE', 'unknown') AS cohort
    FROM applications a JOIN dataset_snapshots s ON a.snapshot_id = s.id
    WHERE s.id = :snapshot_id
)
SELECT source_id, provenance, cohort, COUNT(*) AS applications,
       COUNT(*) FILTER (WHERE target = 1) AS positive_labels,
       AVG(target::numeric) AS observed_label_rate
FROM source_rows GROUP BY source_id, provenance, cohort ORDER BY cohort
