SELECT snapshot_id, source_key, COUNT(*) AS occurrences
FROM applications GROUP BY snapshot_id, source_key HAVING COUNT(*) > 1
