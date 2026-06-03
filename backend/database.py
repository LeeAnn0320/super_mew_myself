import os
from sqlalchemy import create_engine

from sqlalchemy.orm import declarative_base,sessionmaker


DATABASE_URL=os.getenv(
    "DATABASE_URL","postgresql+psycopg2://postgres:postgres@localhost:5432/langchain_app",
)

#langchain_app是数据库名

engine=create_engine(
    DATABASE_URL,pool_pre_ping=True
)



SessionLocal=sessionmaker(bind=engine,autoflush=False,autocommit=False,expire_on_commit=False)


Base=declarative_base()

def init_db()->None:
    import models
    Base.metadata.create_all(bind=engine)

