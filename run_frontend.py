import os
import sys
import logging
from pathlib import Path
import http.server
import socketserver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler that serves index.html for root and unmatched paths"""
    
    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.path = '/index.html'
        
        return super().do_GET()
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        return super().end_headers()


if __name__ == "__main__":
    frontend_path = Path(__file__).parent / "frontend"
    os.chdir(frontend_path)
    
    PORT = 3000
    Handler = CustomHTTPHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            logger.info(f"🌐 Frontend server running at http://localhost:{PORT}")
            logger.info(f"📂 Serving files from: {frontend_path}")
            logger.info("⏹️  Press Ctrl+C to stop")
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info(" Server stopped")
        sys.exit(0)
    except Exception as e:
        logger.error(f" Server error: {e}")
        sys.exit(1)
