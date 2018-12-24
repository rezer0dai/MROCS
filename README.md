# MROCS
benchmark of MROCS algorithm inspired by MADDPG and my previous work ( wheeler ), benchmark for new framework rewheeler ( comming soon ) on multi agent environment Tennis from Unity

Project Details
- state_size=24, action_size=2 as default UnityML - Tennis environment provides
  - however for critic we used state_size=48, action_size=4, as MROCS approach ( update of MADDPG algo )
- 2 players environment, used shared actor + critic for controlling all arms
- Policy Gradients used, namely MROCS algorithm ( DDPG + MADDPG ideas/implementation and updates )
- environment is considered as Udacity asks ~ mean of last 100 episodes during training > .5
  - however this can be far off, as we train TARGET network. Aka i dont think it is reasonable measuring
  - especially when it comes to model itself change every x-timestamps, therefore question is : What we evaluating robustness of model to changes, or his capability to solve given task ? Appareantelly by rubric it is former case
  - Solution ? check at the bottom graphs with TARGET vs EXPLORER. Solution is basically run for evaluation only TARGET network, and environment will be solved much more faster ..
- number of episodes to solve ? based on rubric it is 400, based on real ep executed about 500, based on real capability of target ~ i did not meassure should be 0in between 0~500
- How to install :
  - install anaconda : https://www.anaconda.com/download/
  - then follow : 
  ```
  conda install -y pytorch -c pytorch
  pip install unityagents
  ```
  - then replicate my results by running Report.ipynb
  - and you can download my pretrained weights : https://github.com/rezer0dai/UnityMLEnvs/tree/master/models/mrocs/benchmark
  - drop those checkpoints in main dir, and load agents cells, avoid retraining, and skip to cell which loads those weights and try EXPLORER vs TARGET network
  - environment itself, download and copy to ./data/ folder in this project : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
- if anything more needed in Readme please specify in more detail
