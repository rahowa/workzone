# workzone
REST api setup for industrial robot work zone estimation and control. Include face detection/recognition module, object detection module and zone estimation module. 
# prepare for up in docker
You must to have installed docker and docker-compose.
You can find installtion guide for docker by this [link](https://docs.docker.com/engine/install/ubuntu/) and install docker-compose by the following command - pip install docker-compose
# up in docker
For build or rebuild and up flask and mongo you need use the next one:
    docker-compose up -d --build
For only flask or mongo use:
    docker-compose up -d --build 'service'(example: flaks or mongo)
If you want to see the service status you should use:
    docker-compose ps
If you need to see sevice logs you can use:
    docker-compose logs 'service'(example: flaks or mongo)
Also you can see flask logs at ./logs/server.log 