# Face Recognition

This project aims to implement an advanced face recognition system. The Labeled Faces inthe Wild (LFW), a widely used public dataset in the facial recognition field was used for training. To implement the advanced face recognition technology, we used various image classification methods and compared the performances of each model.

**Dataset** : [The Labeled Faces inthe Wild (LFW)](https://vis-www.cs.umass.edu/lfw/)  (currently inactive)

**Result (Test accuracy)** : 

SIFT - 32.68%   
SIFT with Bag of Visual Words (BOVW) - 33.66%  
SIFT with BOVW and color conversion -  35.63%  

CNN -   
65.20% when individuals with at least 20 images were used.  
67.14% when individuals with at least 30 images were used.  
80.18% when individuals ith at least 40 images were used.  

CNN with more layer, larger image size and more data augmentation -   
57.9934%  
63.4318%  
75.1786%
