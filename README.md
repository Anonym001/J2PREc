# J2PRec
## Introduction 

Joint Product-Participant Recommendation (J2PRec) is a new framework for group-buying (GB) recommendation, which recommends both candidate products and participants for maximizing the success rate of a GB. J2PRec first encodes the various relations in GB for learning enhanced user and product embeddings, and then jointly learns the product and participant recommendation tasks under a probabilistic framework to maximize the GB likelihood.

## Environment Requirement

+ Python == 3.6.5. 
+ tensorflow == 1.14.0
+ numpy == 1.15.4
+ scipy == 1.1.0
+ sklearn == 0.20.0



## Datasets

+ social_relation.txt
   + Introduction: social relations among users. 
   + Format:(<user_id1> \t <user_id2>), denoting there is a social relation between the two users.
+ train_id.txt: 
   + Introduction: GB interactions for training. 
   + Format:(<initiator_id> \t <product_id> \t <participant_id1> \t ... \t <participant_idK>), denoting the initiator purchase the target product with the set of participants. 
+ tune.txt: 
   + Introduction: GB interactions for validation. 
   + Format:(<initiator_id> \t <product_id> \t <participant_id1> \t ... \t <participant_idK>)
+ test.txt:
   + Introduction: GB interactions for testing. 
   + Format:(<initiator_id> \t <product_id> \t <participant_id1> \t ... \t <participant_idK>)
+ test.negative.txt: 
   + Introduction: the sampled negative products for each test user. 
   + Format:(<user_id> \t <product_id1> <product_id2> ... <productz_id1000>)
+ train.txt:
   + Introduction: the user and all her interacted products in training set. 
   + Format:(<user_id> <product_id1> <product_id2> ... <productz_idM>)
+ Due to the file size limit, we provide an illustration example on brightkite dataset. The beibei and gowalla datasets can be downloaded here: https://drive.google.com/drive/folders/1PUUP7mA2xR-suP108jreXyOT-EcjnXBd?usp=sharing. 
   
## Running Command 

+ Data Preparation 

    Please run the data_preprocess.py in the Data folder to generate the required files. 

+ Model Training and Testing

    Please run the following example codes for model training and testing: 

         python3.6 Main.py --dataset brightkite --regs [1e-4,1e-4] --layer_size [64,64] --embed_size 64 --lr 0.001 --epoch 1000 --batch_size 1024 

    You need to specify serveral parameters for training and testing:
   
    + dataset: beibei / brightkite / Yelp
    + regs: regularization weight 
    + layer_size: the number of layers and embedding size 
    + lr: learning rate
    + batch_size : the size of batch for training
    + epoch : the epoch for training 
   
   
   
   
