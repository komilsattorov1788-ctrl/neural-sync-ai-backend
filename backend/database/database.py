import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# PostgreSQL connection string: postgresql+asyncpg://user:pass@host/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./ledger.db")

engine_kwargs = {
    "echo": False,
}
if "sqlite" not in DATABASE_URL:
    engine_kwargs.update({
        "pool_size": 50,
        "max_overflow": 20,
        "pool_pre_ping": True
    })

engine = create_async_engine(DATABASE_URL, **engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
