INSERT INTO feature_snapshots(id, snapshot_id, version, schema, row_count)
SELECT :feature_id, s.id, :version, CAST(:schema AS jsonb), COUNT(a.id)
FROM dataset_snapshots s JOIN applications a ON a.snapshot_id = s.id
WHERE s.id = :snapshot_id GROUP BY s.id
ON CONFLICT(snapshot_id, version) DO NOTHING
