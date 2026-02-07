from fastapi.security import OAuth2PasswordBearer
from fastapi import  Depends, HTTPException, status
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from config import User
import os
from typing import Optional
from pwdlib import PasswordHash
from sqlmodel import Session, select
from db import engine, AdminUser

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
password_hash = PasswordHash.recommended()

def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plaintext password against a hashed password.

    Args:
        plain: The plaintext password provided by the user.
        hashed: The stored password hash to verify against.

    Returns:
        True if the plaintext password matches the hashed password; otherwise False.
    """
    return password_hash.verify(plain, hashed)

def get_password_hash(password):
    """Hash a plaintext password for storage.

    Args:
        password: The plaintext password to hash.

    Returns:
        A hashed password string suitable for storage.
    """
    return password_hash.hash(password)

def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user against the persisted admin user record.

    This function queries the database for the configured admin user record and
    validates both the username and password.

    Args:
        username: Username provided by the client.
        password: Plaintext password provided by the client.

    Returns:
        A User object if authentication succeeds; otherwise None.
    """
    with Session(engine) as session:
        user = session.exec(select(AdminUser)).all()
        if not user:
            return None
        if len(user) != 1:
            return None
        if username != user[0].username:
            return None
        if not verify_password(password, user[0].hashed_password):
            return None
        return User(username=user[0].username)

def create_access_token(subject: str, expires_delta: timedelta) -> str:
    """Create a signed JWT access token.

    The token includes the subject (`sub`), issued-at time (`iat`), and expiry
    time (`exp`), and is signed using the configured secret key and algorithm.

    Args:
        subject: Subject identifier to embed in the token (typically a username).
        expires_delta: Duration after which the token should expire.

    Returns:
        A signed JWT access token as a string.
    """
    now = datetime.now(timezone.utc)
    payload = {"sub": subject, "iat": now, "exp": now + expires_delta}
    return jwt.encode(payload, os.getenv("SECRET_KEY"), algorithm=os.getenv("ALGORITHM"))

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Resolve the current authenticated user from a bearer token.

    This function decodes the JWT access token, extracts the subject claim, and
    returns a corresponding User instance.

    Args:
        token: Bearer token extracted from the Authorization header.

    Returns:
        A User object representing the authenticated principal.

    Raises:
        HTTPException: If the token is invalid, expired, or missing required claims.
    """
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=os.getenv("ALGORITHM"))
        username: str = payload.get("sub")
        if not username:
            raise cred_exc
    except JWTError:
        raise cred_exc
    return User(username=username)