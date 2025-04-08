import os
import shutil
import subprocess
from pathlib import Path

def create_lambda_package():
    # Create a temporary directory for the package
    package_dir = Path('lambda_package')
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    # Copy the Lambda application
    shutil.copy('lambda_app.py', package_dir / 'lambda_app.py')
    
    # Copy the model file
    model_path = Path('runs/train/exp/weights/best.pt')
    if model_path.exists():
        shutil.copy(model_path, package_dir / 'model.pt')
    else:
        print("Warning: Model file not found. Please ensure the model is trained before packaging.")
    
    # Install dependencies
    subprocess.run([
        'pip', 'install',
        '-r', 'requirements.txt',
        '--target', str(package_dir),
        '--platform', 'manylinux2014_x86_64',
        '--only-binary=:all:',
        '--upgrade'
    ])
    
    # Create deployment package
    shutil.make_archive('lambda_deployment', 'zip', package_dir)
    
    # Clean up
    shutil.rmtree(package_dir)
    
    # Print package size
    package_size = os.path.getsize('lambda_deployment.zip') / (1024 * 1024)  # Size in MB
    print(f"\nLambda deployment package created successfully!")
    print(f"Package size: {package_size:.2f} MB")
    print("\nYou can now upload lambda_deployment.zip to AWS Lambda")

if __name__ == '__main__':
    create_lambda_package() 