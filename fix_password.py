
import sys
import os

# Add src to path so we can import ratelmind
# The backend is at /Users/a77/platform/backend
sys.path.insert(0, "/Users/a77/platform/backend/src")

from ratelmind.app import create_app
from ratelmind.extensions import db
from ratelmind.models.user import User

# Initialize app
# We need to make sure we are in the backend dir for .env loading or pass config
# But create_app might look for .env
# Let's change cwd to backend before creating app
os.chdir("/Users/a77/platform/backend")

app = create_app()

with app.app_context():
    print("Searching for user admin@ratelmind.ai...")
    try:
        user = db.session.query(User).filter_by(email="admin@ratelmind.ai").first()
        if user:
            print(f"User found: {user.email}")
            if hasattr(user, 'set_password'):
                user.set_password("changeme123!")
            else:
                user.password = "changeme123!"
                
            db.session.add(user)
            db.session.commit()
            print("Password reset success to: changeme123!")
        else:
            print("User NOT found!")
    except Exception as e:
        print(f"Error: {e}")
