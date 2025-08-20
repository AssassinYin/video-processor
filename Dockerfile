# Use the official PostgreSQL image as a base
FROM postgres:13

COPY populate_db.sh /docker-entrypoint-initdb.d/

# Health check to ensure the database is ready
HEALTHCHECK --interval=5s --timeout=5s --retries=5 \
  CMD pg_isready -U myuser -d mydatabase
