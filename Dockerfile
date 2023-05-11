#FROM ubuntu:latest
FROM jinaai/jina:latest-perf
LABEL authors="Jay"

####
# The /app directory should act as the main application directory
WORKDIR /app

# Copy the app package and package-lock.json file
#COPY package*.json ./

# Copy local directories to the current local directory of our docker image (/app)
COPY ./ ./
#cd /app
#ls -al
# Install node packages, install serve, build the app, and remove dependencies at the end
#RUN apt update -y
#RUN apt install python3
#RUN python3 get-pip.py
#RUN pip install -U Jina
CMD [ "pip", "install", "docarray"]
CMD [ "python", "/app/flow.py"]
#RUN /app/flow.py

EXPOSE 1234

# Start the app using serve command
CMD [ "serve", "-s", "build" ]
####

# Auto generated
#ENTRYPOINT ["top", "-b"]