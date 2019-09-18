0.Use MatterPort's Mask R-CNN Code
1.We start from balloon.py (1 class)
2.We need 2 classes (standing pig,lying pig) 
3.So we refer to https://github.com/SUYEgit/Surgery-Robot-Detection-Segmentation
4.I seperate train,test code.(train.py ,test.py) Because we need understand code.
5.Now we combine these code again(main.py), and make GUI with PyQt5

We change mrcnn/visualize.py  -> display_instances method ( save img file here )
In pig_images/   json file "region_attributes":{"name" : "standing_pig"} <---- class name
We use VGG Image Annotation tool http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html

Not complete project (19/09/18)
