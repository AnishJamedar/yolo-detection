import os

def init_app():
    """Initialize the application by creating necessary directories."""
    # Create uploads directory if it doesn't exist
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print(f"Created directory: {uploads_dir}")
    else:
        print(f"Directory already exists: {uploads_dir}")
    
    # Create a .gitkeep file to ensure the directory is tracked by git
    gitkeep_file = os.path.join(uploads_dir, '.gitkeep')
    if not os.path.exists(gitkeep_file):
        with open(gitkeep_file, 'w') as f:
            f.write('')
        print(f"Created file: {gitkeep_file}")
    else:
        print(f"File already exists: {gitkeep_file}")

if __name__ == '__main__':
    init_app() 