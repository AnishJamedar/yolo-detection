import json
import base64
from pathlib import Path
from lambda_app import lambda_handler

def test_lambda_function(image_path):
    # Read the image file
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Create a test event
    test_event = {
        'body': json.dumps({
            'image': image_data
        })
    }
    
    # Call the Lambda handler
    response = lambda_handler(test_event, None)
    
    # Parse the response
    response_body = json.loads(response['body'])
    
    if response['statusCode'] == 200:
        # Save the output image
        output_image = base64.b64decode(response_body['image'])
        output_path = Path('output.jpg')
        with open(output_path, 'wb') as f:
            f.write(output_image)
        print(f"Success! Output saved to {output_path}")
    else:
        print(f"Error: {response_body.get('error', 'Unknown error')}")

if __name__ == '__main__':
    # Test with a sample image
    test_image = Path('test_image.jpg')
    if test_image.exists():
        test_lambda_function(test_image)
    else:
        print(f"Error: Test image {test_image} not found") 