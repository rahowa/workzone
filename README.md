# workzone

REST api setup for industrial robot work zone estimation and control. Include face detection/recognition module, object detection module and zone estimation module. 

# prepare for up in docker

You must to have installed docker and docker-compose.<br>
You can find installtion guide for docker by this [link](https://docs.docker.com/engine/install/ubuntu/) and install docker-compose by the following command - pip install docker-compose

# up in docker

For build or rebuild and up flask and mongo you need use the next one:<br>
    docker-compose up -d --build<br>
For only flask or mongo use:<br>
    docker-compose up -d --build 'service'(example: flaks or mongo)<br>
If you want to see the service status you should use:<br>
    docker-compose ps<br>
If you need to see sevice logs you can use:<br>
    docker-compose logs 'service'(example: flaks or mongo)<br>
Also you can see flask logs at ./logs/server.log<br>