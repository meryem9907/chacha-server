from sqlmodel import SQLModel, create_engine, Field, Session, select
import os
from dotenv import load_dotenv

load_dotenv(".env")

sqlite_url = os.getenv("DATABASE_URL")

class AdminUser(SQLModel, table=True):
    """SQLModel table definition for an administrative user.

    Attributes:
        id: Primary key identifier for the admin user.
        username: Username for the admin user (indexed).
        hashed_password: Password hash for the admin user.
    """
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True)
    hashed_password: str 

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=True, connect_args=connect_args)

def create_db_and_tables():
    """Create database tables defined in SQLModel metadata.

    Uses the module-level SQLAlchemy engine to create all tables declared via
    SQLModel models.
    """
    SQLModel.metadata.create_all(engine)

def get_admin_by_username(username: str) -> AdminUser | None:
    """Fetch an admin user by username.

    Args:
        username: Username to search for.

    Returns:
        The matching AdminUser if found; otherwise None.
    """
    with Session(engine) as session:
        statement = select(AdminUser).where(AdminUser.username == username)
        return session.exec(statement).first()

def create_admin_if_not_exists(admin_user: AdminUser):
    """Create an admin user record if it does not already exist.

    This function checks for an existing admin user with the same username.
    If one exists, no action is taken. Otherwise, the provided AdminUser is
    inserted and committed.

    Args:
        admin_user: The AdminUser instance to create if missing.

    Returns:
        None
    """
    with Session(engine) as session:
        existing = session.exec(
            select(AdminUser).where(AdminUser.username == admin_user.username)
        ).first()

        if existing:
            return  

        session.add(admin_user)
        session.commit()
