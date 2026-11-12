from alembic import context

from fleetrisk.config import Settings
from fleetrisk.data.db import make_engine

if context.is_offline_mode():
    context.configure(url=Settings().database_url, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = make_engine(Settings())
    with engine.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
