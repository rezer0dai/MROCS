# MROCS
benchmark of MROCS algorithm inspired by MADDPG and my previous work ( wheeler ), benchmark for new framework rewheeler ( comming soon ) on multi agent environment Tennis from Unity

Project Details
- state_size=24, action_size=2 as default UnityML - Tennis environment provides
  - however for critic we used state_size=48, action_size=4, as MROCS approach ( update of MADDPG algo )
- 2 players environment, used shared actor + critic for controlling all arms
- Policy Gradients used, namely MROCS algorithm ( DDPG + MADDPG ideas/implementation and updates )
- How to install :
  - install anaconda : https://www.anaconda.com/download/
  - then follow : 
  ```
  conda install -y pytorch -c pytorch
  pip install unityagents
  ```
  - then replicate my results by running MROCS.ipynb
  - and you can download my pretrained weights : https://github.com/rezer0dai/UnityMLEnvs/tree/master/models/mrocs/benchmark
  - drop those checkpoints in main dir, and load agents cells, avoid retraining, and skip to cell which loads those weights and try EXPLORER vs TARGET network
  - environment itself, download and copy to ./data/ folder in this project : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
