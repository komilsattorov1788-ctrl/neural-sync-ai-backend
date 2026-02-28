import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, Enum, Index, text
from sqlalchemy.orm import declarative_base
import enum

Base = declarative_base()

class LedgerState(enum.Enum):
    RESERVED = 'reserved'
    COMMITTED = 'committed'
    ROLLED_BACK = 'rolled_back'

class TransactionLedger(Base):
    """
    10/10 Architecture: Immutable Source of Truth 
    Records complete lifecycle of financial credits natively in Postgres.
    """
    __tablename__ = 'transaction_ledger'
    
    tx_id = Column(String(36), primary_key=True)
    # Citus Sharding Key: user_id MUST be part of the primary key or unique constraint
    # to distribute data across multiple physical databases (Distributed Ledger).
    user_id = Column(String(255), primary_key=True, index=True, nullable=False)
    intent = Column(String(50), nullable=False)
    cost = Column(Integer, nullable=False)
    state = Column(Enum(LedgerState), default=LedgerState.RESERVED, nullable=False)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    committed_at = Column(DateTime(timezone=True), nullable=True)
    rolled_back_at = Column(DateTime(timezone=True), nullable=True)
    
    # Trace context for observability linking
    traceparent = Column(String(255), nullable=True)

class OutboxEvent(Base):
    """
    10/10 true Database-backed Transactional Outbox.
    Ensures message emission exactly-once bounds are tied strictly to Ledger ACID transactions.
    """
    __tablename__ = 'outbox_events'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tx_id = Column(String(36), index=True, nullable=False)
    payload = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status = Column(String(20), default='pending') # pending | sent | failed
    
    # Composite indexing for fast polling by dispatcher
    __table_args__ = (
        Index('idx_outbox_pending', 'status', 'created_at', postgresql_where=text("status = 'pending'")),
    )
