# workzone
REST api setup for industrial robot work zone estimation and control. Include face detection/recognition module, object detection module and zone estimation module. 

## Features

## Setup
### Install from source
> Clone repo and recursively update submodules 

### Download weights
- Download weight from <a href="https://drive.google.com/drive/folders/17rK1H2gS6vDzkk14IVQgXaYMP6thf9tH?usp=sharing" >Google Drive</a>
- Move weights to the 'weights' folder inside parent folder 


### Setup database with docker
MongoDB used to store face descriptors for face recognition subsystem
> Download MongoDB docker image
```shell script
$ docker run --name mongo -d mongo:tag
```
> And then start it
``` shell script
$ docker run -d -p 27017:27017 mongo
```

### Start server
> To start server use Flask CLI
```shell script
$ flask run
```
> or use python
```shell script
$ python run.py
```

## Setup robot workzone estimation subsystem
> First update submodules
```shell script
$ git submodule update --init --recursive
```
> Move to `robot_work_zone_estimation` directory
```shell script
$ cd robot_work_zone_estimation
```
> Follow <a href="https://github.com/rahowa/robot_work_zone_estimation/blob/13e50a8bef95817514454f4dc1c42b3d7956c91d/README.md#start">instructions</a> from submodule README.md

## Setup face recognition subsystem
- Provide faces for face recognition subsystem
> Use the following directory structure:
  
    ├── workzone                        # Parent directory
        ├── face_database               # Directory with all persons
        │   ├── person_name_1           # Each person should have it's own folder
        │   │    ├── img1.jpg       
        │   │    ├── img2.jpg
        │   │    ├── img3.jpg
        │   │    └── ...
        │   ├── person_name_2           
        │   │    ├── img1.jpg
        │   │    ├── img2.jpg
        │   │    ├── img3.jpg
        │   │    └── ...
        │   └── ...
        └── ...
- Start server 
- Route to https://127.0.0.1:5000/fill_db
- Done


## Available routes

## Scenarios
### Start scenario
### Add new scenarios

## License