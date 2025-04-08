# YOLO Object Detection Web App

A web application that uses YOLO (You Only Look Once) to detect objects in images and draw circles around them.

## Features

- Upload images for object detection
- Automatic object detection using YOLOv8
- Visual representation with circles drawn around detected objects
- Confidence scores for each detection
- Clean, responsive user interface

## Local Development

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python app.py`
6. Open your browser and navigate to `http://localhost:5000`

## Deployment to Render.com

1. Create a GitHub repository and push this code to it
2. Sign up for a free account on [Render.com](https://render.com)
3. Click "New" and select "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - Name: Choose a name for your service
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Plan: Free
6. Click "Create Web Service"

## Deployment to Heroku

1. Create a GitHub repository and push this code to it
2. Sign up for a free account on [Heroku](https://heroku.com)
3. Create a new app
4. Connect your GitHub repository
5. Enable automatic deploys
6. Add the following buildpacks:
   - heroku/python
   - heroku-community/apt
7. Set the following config vars:
   - `PYTHON_VERSION`: 3.9.0
8. Deploy the app

## Deployment to AWS Elastic Beanstalk

1. Create a GitHub repository and push this code to it
2. Sign up for an AWS account
3. Go to Elastic Beanstalk console
4. Create a new application
5. Choose "Web server environment"
6. Upload your code or connect to your GitHub repository
7. Configure the environment:
   - Platform: Python
   - Python version: 3.9
   - Application code: Upload your code
8. Create the environment

## License

MIT 