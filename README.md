# MROCS
benchmark of MROCS algorithm inspired by MADDPG and my previous work ( wheeler ), benchmark for new framework rewheeler ( comming soon ) on multi agent environment Tennis from Unity

all informations are provided in Report.ipynb
- used own MROCS algorithm inspired by MADDPG
- well there are many improvements ~ see wheeler framework for example ( lot of different features and things to tune ), rewheeler soon

- How to run the code : run Report.ipynb

- details : 
- for actor i used NoisyNetworks for exploration
- i used batchnorm ( 24 x 1 but can be used with same effect 8 x 3 due to the fact 3 frames are stacked )
- network sizes 400 + 300, critic used simple FF encoding for state + action before cat, learning rates see Report.ipynb as well as all hyperparameters ( no need to write them duplicitly here .. )
- MROCS because i used different methodology than MADDPG ( once critic multiple rewards ), detached heads of noisy network for actor ( detached sigma + noise, main model is shared )


trained weights you can download (!!) at : https://github.com/rezer0dai/UnityMLEnvs/tree/master/models/mrocs/benchmark or alternative : https://github.com/rezer0dai/temporary

