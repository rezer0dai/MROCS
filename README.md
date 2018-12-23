# MROCS
benchmark of MROCS algorithm inspired by MADDPG and my previous work ( wheeler ), benchmark for new framework rewheeler ( comming soon ) on multi agent environment Tennis from Unity

all informations are provided in Report.ipynb
- used own MROCS algorithm inspired by MADDPG
- well there are many improvements ~ see wheeler framework for example ( lot of different features and things to tune ~ GAE, RNN/CNN, globalnormalizer, network sizes, neural network architecture, RUDDER approah, etc etc), rewheeler soon

- How to run the code : run Report.ipynb

- details : 
- for actor i used NoisyNetworks for exploration
- i used batchnorm ( 24 x 1 but can be used with same effect 8 x 3 due to the fact 3 frames are stacked )
- network sizes 400 + 300, critic used simple FF encoding for state + action before cat, learning rates see Report.ipynb as well as all hyperparameters ( no need to write them duplicitly here .. )
- updating network every 100(*2 )steps and repeat learning 200 times, postponed update for 1 learnings ( bigger steps i do postpone everytime )
- used 3-step estimator, and Advantage as TD-learning, batch size 256, and buffer size 1e6, rerandomizing noisy nets for explooration every 3rd step
- one detached actor, one critic, small buffer size 3e4 and batch size of 256

- MROCS because i used different methodology than MADDPG ( once critic multiple rewards ), detached heads of noisy network for actor ( detached sigma + noise, main model is shared )


trained weights you can download (!!) at : https://github.com/rezer0dai/UnityMLEnvs/tree/master/models/mrocs/benchmark or alternative : https://github.com/rezer0dai/temporary

