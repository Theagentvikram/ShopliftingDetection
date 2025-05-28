import uvicorn
import os

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the current directory to ensure imports work correctly
    os.chdir(current_dir)
    
    # Run the server with explicit app import path and a different port
    uvicorn.run("main:app", host="0.0.0.0", port=8111, reload=True)