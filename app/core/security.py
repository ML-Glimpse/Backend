"""Authentication and security utilities"""
import bcrypt


def hash_password(password: str) -> str:
    """
    Hash a password for storing using bcrypt

    Args:
        password: Plain text password

    Returns:
        Bcrypt hashed password as string
    """
    # Convert password to bytes (bcrypt requires bytes)
    # Truncate to 72 bytes as bcrypt has this limit
    password_bytes = password.encode('utf-8')[:72]

    # Generate salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)

    # Return as string for storage in MongoDB
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a stored password against a provided password using bcrypt

    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored bcrypt hash

    Returns:
        True if password matches, False otherwise
    """
    try:
        # Convert password to bytes and truncate to 72 bytes
        password_bytes = plain_password.encode('utf-8')[:72]

        # Convert stored hash to bytes
        hashed_bytes = hashed_password.encode('utf-8')

        # Verify password
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        # Return False for any verification errors
        return False
