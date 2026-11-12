CREATE TABLE data_sources (
    id text PRIMARY KEY,
    description text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE ingestions (
    id uuid PRIMARY KEY,
    source_id text NOT NULL REFERENCES data_sources(id),
    relative_path text NOT NULL,
    source_sha256 text NOT NULL CHECK (length(source_sha256) = 64),
    contract_version text NOT NULL,
    provenance text NOT NULL CHECK (provenance IN ('real','synthetic')),
    status text NOT NULL CHECK (status IN ('running','completed','duplicate','failed')),
    snapshot_id text,
    quality jsonb NOT NULL DEFAULT '{}',
    error_code text,
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    replay_of uuid REFERENCES ingestions(id)
);
CREATE INDEX ingestions_source_idx ON ingestions(source_id, created_at DESC);
CREATE TABLE dataset_snapshots (
    id text PRIMARY KEY CHECK (length(id) = 64),
    source_id text NOT NULL REFERENCES data_sources(id),
    source_sha256 text NOT NULL,
    contract_version text NOT NULL,
    provenance text NOT NULL CHECK (provenance IN ('real','synthetic')),
    row_count bigint NOT NULL CHECK (row_count > 0),
    parquet_path text NOT NULL,
    parquet_sha256 text NOT NULL,
    source_manifest jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_id, source_sha256, contract_version, provenance)
);
ALTER TABLE ingestions ADD CONSTRAINT ingestion_snapshot_fk FOREIGN KEY(snapshot_id) REFERENCES dataset_snapshots(id);
CREATE TABLE applications (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    snapshot_id text NOT NULL REFERENCES dataset_snapshots(id),
    source_key text NOT NULL,
    target smallint CHECK (target IN (0,1)),
    attributes jsonb NOT NULL,
    UNIQUE(snapshot_id, source_key)
);
CREATE INDEX applications_snapshot_idx ON applications(snapshot_id, target);
CREATE TABLE loan_observations (
    application_id bigint PRIMARY KEY REFERENCES applications(id),
    amount numeric(20,2) CHECK (amount >= 0),
    asset_value numeric(20,2) CHECK (asset_value >= 0),
    currency char(3),
    observation_stage text NOT NULL DEFAULT 'source_snapshot',
    CHECK (currency IS NULL OR currency ~ '^[A-Z]{3}$')
);
CREATE TABLE historical_credit_rows (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    snapshot_id text NOT NULL REFERENCES dataset_snapshots(id),
    source_key text NOT NULL,
    application_key text NOT NULL,
    previous_key text,
    relative_day integer,
    attributes jsonb NOT NULL,
    UNIQUE(snapshot_id, source_key)
);
CREATE INDEX historical_application_idx ON historical_credit_rows(snapshot_id, application_key);
CREATE TABLE quality_results (
    ingestion_id uuid PRIMARY KEY REFERENCES ingestions(id),
    result jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE feature_snapshots (
    id text PRIMARY KEY,
    snapshot_id text NOT NULL REFERENCES dataset_snapshots(id),
    version text NOT NULL,
    schema jsonb NOT NULL,
    row_count bigint NOT NULL CHECK(row_count > 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(snapshot_id, version)
);
CREATE TABLE experiments (
    id uuid PRIMARY KEY,
    snapshot_id text NOT NULL REFERENCES dataset_snapshots(id),
    feature_snapshot_id text NOT NULL REFERENCES feature_snapshots(id),
    metadata jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE model_versions (
    id uuid PRIMARY KEY,
    experiment_id uuid NOT NULL REFERENCES experiments(id),
    version text NOT NULL,
    target text NOT NULL,
    horizon text NOT NULL,
    provenance text NOT NULL CHECK(provenance IN ('real','synthetic')),
    artifact_path text NOT NULL UNIQUE,
    manifest jsonb NOT NULL,
    validated boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX models_experiment_idx ON model_versions(experiment_id);
CREATE TABLE predictions (
    id uuid PRIMARY KEY,
    model_id uuid NOT NULL REFERENCES model_versions(id),
    application_key text NOT NULL,
    dataset_id text NOT NULL REFERENCES data_sources(id),
    provenance text NOT NULL CHECK(provenance IN ('real','synthetic')),
    raw_probability double precision NOT NULL CHECK(raw_probability >= 0 AND raw_probability <= 1),
    calibrated_probability double precision CHECK(calibrated_probability >= 0 AND calibrated_probability <= 1),
    risk_band text NOT NULL CHECK(risk_band IN ('lower','moderate','higher')),
    feature_schema_version text NOT NULL,
    feature_sha256 text NOT NULL,
    policy_version text NOT NULL,
    request_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX predictions_latest_idx ON predictions(model_id, dataset_id, provenance, application_key, created_at DESC, id DESC);
CREATE TABLE financial_assumptions (
    id uuid PRIMARY KEY,
    definition jsonb NOT NULL,
    result jsonb NOT NULL,
    request_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE audit_events (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_type text NOT NULL,
    entity_id text NOT NULL,
    request_id text,
    metadata jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX audit_entity_idx ON audit_events(entity_id, created_at DESC);
