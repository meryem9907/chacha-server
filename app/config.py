from pydantic import BaseModel

class Token(BaseModel):
    """
    Authentication token response model.

    Attributes:
        access_token: Encoded access token string used for authenticated requests.
        token_type: Token type identifier (defaults to "bearer").
    """
    access_token: str
    token_type: str = "bearer"

class User(BaseModel):
    """
    User identity model.

    Attributes:
        username: Unique username identifying the user.
    """
    username: str
