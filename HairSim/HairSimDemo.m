% This code will generate a simulated hair-occluded image by corrupting a hair-free dermoscopic image. 
% 
% Run Main.m to see a demo of the code.   
%
% HairSim by Hengameh Mirzaalian is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License 
% (http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US).



clc
close all
clear all

ImagePath.Img_hair_free='./Figs/011ch1.tif';    %Hair-free dermoscopic image
ImagePath.Img_hair_occluded='./Figs/024pa2';%Hair-occluded dermoscopic image
ImagePath.Simulated_Img='./Figs/PhantImg'; 
ImagePath.Hair_Mask='./Figs/Mask'; 



disp('This is a demo of the HairSim software.')
disp('  ')
disp('The input is a hair free dermoscopic image and as the output HairSim provides you a hair occluded  dermoscopic image accompanied with its hair mask.')
disp('  ')
disp('HairSim can be run in different modes (mode#1,2,3). ')
disp('  ')
flag=1;
while flag
 mode = input('Type a number (1, or 2, or 3)  to start with that mode --> mode=', 's');
disp('  ')

if ~isstrprop(mode, 'digit')
    disp('Please select a number (1, 2, or 3), not a character! ')
    %mode = input(' mode=', 's');
    disp('  ')
else
    mode=str2num(mode(1));
end
if (mode~=1 &  mode~=2 &  mode~=3)
    disp('Please select a number (1, 2, or 3)! ')
    %mode = input(' mode=', 's');
    disp('  ')
else
    
    flag=0;
    switch mode

        case 1
    


clc
disp('TEST MODE 1:  Example 1.1:  Generating random straight hairs using random color. ')

%-----------------------
% TEST MODE 1:
%-----------------------
% Example 1.1:
Parameters.mode=1;
Parameters.Colors=[];
Parameters.sigma=[];
Parameters.Thickness=[];
Parameters.Curliness=0;%Straight hairs can be generated by setting curliness to 0;
[Simulated_Img11,Hair_Mask11] = HairSim(ImagePath,Parameters);
% Example 1.2:
clc
disp('TEST MODE 1:  Example 1.2:  Generating random  hairs (straight/curly) using random colors. ')
M=7;
Parameters.mode=1;
Parameters.Colors=rand(3,M);
Parameters.sigma=randi(10,1,M);
Parameters.Thickness=randi(10,1,M);
Parameters.Curliness=ones(1,10);
ImagePath.Img_hair_occluded=[];
[Simulated_Img12,Hair_Mask12] = HairSim(ImagePath,Parameters);

clc
disp('TEST MODE 1:  Example 1.3:  Generating random  (straight/curly)  hairs using predefined colors. ')
% Example 1.3:
Parameters.mode=1;
Parameters.Colors=[0.6824 0.4863 0.3137;0.4549    0.2392    0.1804 ; 0.5412 0.2314 0.1804];
Parameters.sigma=[5 10];
Parameters.Thickness=[1 7];
Parameters.Curliness=[.5 1];
ImagePath.Img_hair_occluded=[];
[Simulated_Img13,Hair_Mask13] = HairSim(ImagePath,Parameters);

 case 2

%-----------------------
% TEST MODE 2:
%-----------------------
% Example 2.1:
clc
disp('TEST MODE 2:  Manually define hair shaft using sets of points on a hair-free dermoscopic image. ')
M=7;
Parameters.mode=2;
Parameters.Colors=[];
Parameters.sigma=5;
Parameters.Thickness=3;
Parameters.Curliness=rand(1,M);
ImagePath.Img_hair_occluded=[];
[Simulated_Img2,Hair_Mask2] = HairSim(ImagePath,Parameters);


 case 3
%-----------------------
% TEST MODE 3:
%-----------------------
% Example 3.1:
clc
disp('TEST MODE 3:   Manually trace the hair shafts of a hair-occluded dermoscopic image.  ')
Parameters.mode=3;
Parameters.Thickness=3;
Parameters.Colors=[];
Parameters.sigma=5;
Parameters.Curliness=0;
ImagePath.Img_hair_occluded='./Figs/024pa2.tif';%Hair-occluded dermoscopic image
[Simulated_Img3,Hair_Mask3] = HairSim(ImagePath,Parameters);

    end
end
end
