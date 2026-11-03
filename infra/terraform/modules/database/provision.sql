-- RDS counterpart of infra/postgres/init/01-databases.sql, run by the db-provision
-- task as the RDS master user. Passwords come from Secrets Manager as psql variables.
-- Repeatable: existing roles take their current password. The master user is not a
-- superuser. Since PostgreSQL 16, creating a role grants the creator only the ADMIN
-- option, without SET, so the master user grants itself SET and INHERIT before it
-- creates databases owned by the roles and changes their default privileges.
\set ON_ERROR_STOP on
SELECT pg_advisory_lock(7348922);

SELECT format(
    CASE WHEN EXISTS (SELECT FROM pg_roles WHERE rolname = name)
        THEN 'ALTER ROLE %I LOGIN PASSWORD %L'
        ELSE 'CREATE ROLE %I LOGIN PASSWORD %L'
    END, name, password)
FROM (VALUES
    ('source_app', :'source_app_password'),
    ('source_reader', :'source_reader_password'),
    ('platform_app', :'platform_app_password'),
    ('mlflow_app', :'mlflow_app_password')
) AS roles(name, password)
\gexec

SELECT format('GRANT %I TO CURRENT_USER WITH INHERIT TRUE, SET TRUE', name)
FROM (VALUES ('source_app'), ('source_reader'), ('platform_app'), ('mlflow_app'))
    AS roles(name)
WHERE NOT pg_has_role(CURRENT_USER, name, 'SET')
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
