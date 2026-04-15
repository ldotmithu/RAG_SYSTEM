
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
login(token=os.getenv("HF_API_KEY"))


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from dotenv import load_dotenv
env_file = Path(__file__).parent / ".env"
load_dotenv(env_file)

os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '1800'  # 30 minutes
os.environ['HF_HUB_OFFLINE'] = 'False'
os.environ['REQUESTS_TIMEOUT'] = '1800'

import socket
socket.setdefaulttimeout(300) 

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check environment
required_env = ['GROQ_API_KEY']
missing_env = [var for var in required_env if not os.getenv(var)]

if missing_env:
    logger.error(f"❌ Missing environment variables: {', '.join(missing_env)}")
    logger.info(f"Please add to .env file or set: export GROQ_API_KEY=<your-key-here>")
    sys.exit(1)

logger.info("✅ Environment validated")

# Import and start server
try:
    import uvicorn
    from backend.main import app
    
    logger.info("🚀 Starting AI Backend...")
    logger.info("📍 API available at: http://localhost:8000")
    logger.info("📚 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    
except Exception as e:
    logger.error(f"❌ Failed to start server: {str(e)}")
    sys.exit(1)
