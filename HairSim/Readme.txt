HairSim will generate a simulated hair-occluded image by corrupting a hair-free dermoscopic image. 

HairSim by Hengameh Mirzaalian is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License (http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US).

Run HairSimDemo.m to see a demo of the code.  



Parameters needed to be set:
---------------------------------------------

ImagePath.Img_hair_free         - Path of a hair-free dermoscopic image. 


ImagePath.Img_hair_occluded     - Path of a hair-occluded image (optional).
                                  This path should be set if you run the code in the third mode
                                  (different modes are explained in the following).


ImagePath.Simulated_Img         - Path to save generated hair occluded image.


ImagePath.Hair_Mask             - Path to save hair-mask of the generated hair occluded image.


Parameters.mode                 - Determines the different modes to generate  the medial curves of the hair shafts:                           
                                  Parameters.mode=1 --> Automatic random curve synthesizer. 
                                  Parameters.mode=2 --> Manually define hair shaft using 
                                                        sets of points on a hair-free dermoscopic image.
                                  Parameters.mode=3 --> Manually trace the hair shafts of a hair-occluded 
                                                        dermoscopic image. This mode needs to specify a
                                                        hair-occluded image, e.g.:
                                                        Parameters.Img_hair_occluded=Img_hair_occluded; 

 
                    
Parameters.Colors              - A Mx3  Matrix, where M is a scalar. Using this parameter you can determine the color(s) of the generated hairs. 
                                 This Mx3 matrix is randomly sampled to specify the color of the generated hair. 
                                 Default: Parameters.Colors=
                                                 [0.5412    0.2314    0.1804
                                                  0.6824    0.4863    0.3137
                                                  0.4549    0.2392    0.1804
                                                  0.4400    0.2100    0.1600];


Parameters.sigma               - A 1xM matrix to determine the standard deviation of the Gaussian function (in pixels) used for smoothing the generated hairs, which is randomly sampled. 
                                 Default: a random 8x3 matrix: 1+randi(10,1,8);

  
Parameters.Thickness           - A 1xM matrix to determine the thickness value at the centre of the hair (the thickest part of the hair). 
                                 This 1xM matrix is randomly sampled to specify the thickness of the generated hair. 
                                 Default: a random 1x8 matrix: 1+randi(10,1,8).

 
Parameters.Curliness           - Determine the curliness of the hairs, which is randomly sampled. 
                                 It can take values between [0,1]. 
                                 Default: a random 1x8 matrix: randi(1,8)





