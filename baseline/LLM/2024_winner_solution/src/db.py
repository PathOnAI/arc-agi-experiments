import os

import asyncpg

from src import logfire

pool: asyncpg.pool.Pool | None = None


async def init_db_pool():
    global pool
    pool = await asyncpg.create_pool(dsn=os.environ["NEON_DB_DSN"])
    logfire.debug("Database connection pool created.")


async def close_db_pool():
    global pool
    if pool:
        await pool.close()
        logfire.debug("Database connection pool closed.")
