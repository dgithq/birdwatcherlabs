# birdwatcherlabs.com
## Easy Webserver Configuration
### Get Docker image
Download the image from Docker Hub:

`docker pull dqiao235/webserver:latest`

### Run the image

`docker run -p 80:5000 -d webserver:latest`

Now, your local host is listening on port 80 for HTTP requests, and will forward traffic to port 5000 of the container, which will handle the requests.

