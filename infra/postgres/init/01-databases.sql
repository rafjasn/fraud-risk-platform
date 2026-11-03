-- Local development credentials. Run with psql as the cluster administrator.
-- Reused by the image's first-start hook and the repeatable db-provision job.
\set ON_ERROR_STOP on
SELECT pg_advisory_lock(7348922);

SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', name, password)
FROM (VALUES
    ('source_app', 'local-source'),
    ('source_reader', 'local-source-reader'),
    ('platform_app', 'local-platform'),
    ('mlflow_app', 'local-mlflow')
) AS roles(name, password)
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = name)
\gexec

-- CREATE DATABASE cannot run inside a transaction or a DO block.
SELECT format('CREATE DATABASE %I OWNER %I', name, owner)
FROM (VALUES
    ('fraud_source', 'source_app'),
    ('fraud_platform', 'platform_app'),
    ('mlflow', 'mlflow_app')
) AS databases(name, owner)
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = name)
\gexec

REVOKE ALL ON DATABASE fraud_source FROM PUBLIC;
REVOKE ALL ON DATABASE fraud_platform FROM PUBLIC;
REVOKE ALL ON DATABASE mlflow FROM PUBLIC;
GRANT CONNECT ON DATABASE fraud_source TO source_reader;

SELECT pg_advisory_unlock(7348922);

\connect fraud_source
GRANT USAGE ON SCHEMA public TO source_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO source_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE source_app IN SCHEMA public
    GRANT SELECT ON TABLES TO source_reader;
