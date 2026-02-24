# main.py at root level
import uvicorn
import os

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.app:app", host="127.0.0.1", port=port, reload=False)