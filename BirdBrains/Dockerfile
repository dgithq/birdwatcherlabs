# Use the official image as a parent image.
FROM pytorch/pytorch:latest
ENV FLASK_APP=hello

# Set the working directory.
WORKDIR /usr/src/app

RUN pip install pandas && pip install Flask

# Copy the file from your host to your current location.
COPY . .


# Inform Docker that the container is listening on the specified port at runtime.
EXPOSE 5000

# Run the specified command within the container.
CMD [ "flask", "run", "--host=0.0.0.0" ]


