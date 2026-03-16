
import sys
import os

# Add src to path so we can import ratelmind
sys.path.insert(0, "/Users/a77/platform/backend/src")

from ratelmind.app import create_app
from ratelmind.extensions import db
from ratelmind.models.user import User, UserRole
from ratelmind.models.tenant import Tenant

# Change cwd to backend
os.chdir("/Users/a77/platform/backend")

app = create_app()

TENANT_ID = "a86d92b1-0aa3-45c4-8662-034ec05add0b"
EMAIL = "gaoqi@ratelmind.ai"
PASSWORD = "yourpassword"

with app.app_context():
    print(f"Creating tenant admin for tenant {TENANT_ID}...")
    
    # Check if tenant exists
    tenant = db.session.get(Tenant, TENANT_ID)
    if not tenant:
        print("Tenant not found!")
        sys.exit(1)
        
    # Check if user exists
    existing = db.session.query(User).filter_by(email=EMAIL).first()
    if existing:
        print("User already exists!")
        sys.exit(0)
        
    user = User(
        email=EMAIL,
        password=PASSWORD, # Model handles hashing if property setter exists
        tenant_id=TENANT_ID,
        role=UserRole.TENANT_ADMIN.value,
        first_name="Gao",
        last_name="Qi"
    )
    
    # Just in case direct assignment didn't hash it (depends on model implementation)
    if hasattr(user, 'set_password'):
        user.set_password(PASSWORD)
        
    user.verify_email()
    
    db.session.add(user)
    db.session.commit()
    print(f"User {EMAIL} created successfully!")
