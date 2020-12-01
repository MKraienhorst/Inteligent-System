function [PhantImg,Mask]=HairSimulator(ImagePath,Parameters)
% This code will generate a simulated hair-occluded image by corrupting a hair-free dermoscopic image.
%
% Run Main.m to see a demo of the code.
%
% Parametes need to be set as input:
% ---------------------------------------------------
%
% ImagePath.Img_hair_free               - Path of a hair-free dermoscopic image.
%
% ImagePath.Img_hair_occluded     - Path of a hair-occluded image (optional).
%                                                                 This path should be set if you run the code in the third mode.
%
% ImagePath.Simulated_Img            - Path to save the generated hair occluded image.
%
% ImagePath.Hair_Mask                     - Path to save the hair-mask of the generated hair occluded image.
%
% Parameters.mode                             - Determine different modes to generate  the medial curves of the hair shafts:
%                                                                  Parameters.mode=1 --> Automatic random curve synthesizer.
%                                                                  Parameters.mode=2 --> Manually define hair shaft using
%                                                                                                                sets of points on a hair-free dermoscopic image.
%                                                                 Parameters.mode=3 --> Manually trace the hair shafts of a hair-occluded
%                                                                                                               dermoscopic image. This mode needs to specify a
%                                                                                                                hair-occluded image, e.g.:
%                                                                  Parameters.Img_hair_occluded=Img_hair_occluded;
%
%
% Parameters.Colors                            - A Mx3  Matrix, where M is a scalar. Using this parameter you can determine
%                                                                    the color/s of the generated hairs.
%                                                                    This Mx3 matrix is randomly sampled to specify the color of the generated hair.
%                                                                    Default: Parameters.Colors=    [0.5412    0.2314    0.1804
%                                                                                                                               0.6824    0.4863    0.3137
%                                                                                                                               0.4549    0.2392    0.1804
%                                                                                                                               0.4400    0.2100    0.1600];
%
% Parameters.sigma                              - A 1xM matrix to determine the standard deviation of the Gaussian function (in pixels)
%                                                                    used for smoothening the generated hairs, which is randomly sampled.
%                                                                    Default: a random 8x3 matrix: 1+randi(10,1,8);
%
% Parameters.Thickness                      - A 1xM matrix to determine the thickness value at the centre of the hair
%                                                                     (the thickestpart of the hair).
%                                                                  This 1xM matrix is randomly sampled to specify the thickness of the generated hair.
%                                                                   Default: a random 1x8 matrix: 1+randi(10,1,8).
%
% Parameters.Curliness                        - Determine the curliness of the hairs, which is randomly sampled.
%                                                                     It can take values between [0,1]. 
%                                                                     Default: a random 1x8 matrix: randi(1,8)
%
% HairSim by Hengameh Mirzaalian is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License 
% (http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US).



close all;
Img_hair_free=imread(ImagePath.Img_hair_free);
% The following functions:
%   - bspline_wdeboor
%   - bspline_deboor
% are used from the B-splines package  by Levente Hunyadi.

[M,N,z]=size(Img_hair_free);
PhantImg=Img_hair_free;
fig=figure (1);
h=gcf;set(h,'Position',[228 372 676 572])
[Parameters]=parameter_Initialization(Img_hair_free,Parameters,ImagePath,[M N]);
PhantImg=Img_hair_free;
Mask=zeros(M,N);
flag=1;
debug_save_results=0;
counter=0;
while flag
    figure (1);
    counter=counter+1;
    if Parameters.mode==1
        [x0,y0]=GenPntsMode1(Parameters,M,N);
    else
        disp('  ')
        disp(Parameters.generation_mode)
        disp('Use double-click or enter after adding the final point.')
        flag_num_pnts=1;
        while flag_num_pnts & flag
            [y0, x0]=getpts;
            if size(x0,1)>3
                flag_num_pnts=0;
            else
                disp('    ')
                disp('You need to select at least 4 points on Fig1!')
                disp('and after adding the final point, use double-click or enter .')
                x0=[];
%                 Next_Hair=input('Press enter to generate a new hair or type `n’ to stop!    ','s');
%                 if isempty(Next_Hair)
%                     Next_Hair = 'y';
%                 end
%                 if ~strcmp(Next_Hair,'y')
%                     flag=0;
%                 end
                
            end
        end
    end
    disp( '  ')
    if   (size(x0,1)>0)
        [hair_Skel,XX,YY,FlagcheckOutOfRangeSubscript]=hairMask(x0,y0,M,N);
        debug_save_results=1;
        if ~FlagcheckOutOfRangeSubscript
            mask=HairDilation(XX,YY,M,N,hair_Skel,Parameters.Thickness(randi(size(Parameters.Thickness,2))));
            %             clr=Parameters.Colors(:,randi(size(Parameters.Colors,2)))';
            clr=Parameters.Colors(randi(size(Parameters.Colors,1)),:);
            PhantImg=ColorFulHair(mask,PhantImg,clr,Parameters.sigma(randi(size(Parameters.sigma,2))));
            Mask=(Mask+1-mask)>0;
        end
        figure (2); h=gcf;set(h,'Position',[918 372 656 572])
        
        subplot(131);imshow(Img_hair_free);title('Hair free dermoscopic image')
        subplot(132);imshow(Mask);title('Hair-Mask')
        subplot(133);imshow(PhantImg);title('Simulated hair occluded image')
        %clc
        Next_Hair=input('Press enter to generate a new hair or type `n’ to stop!    ','s');
        if isempty(Next_Hair)
            Next_Hair = 'y';
        end
        if ~strcmp(Next_Hair,'y')
            flag=0;
        end
    end
end
if debug_save_results
    save([ImagePath.Hair_Mask '.mat']','Mask')
    save([ImagePath.Simulated_Img '.mat'],'PhantImg')
end
end
function  [x0,y0]=GenPntsMode1(Parameters,M,N)
x0=1+fix((M-1)*rand(7,1));
y0=1+fix((N-1)*rand(7,1));
D=sqrt((x0(1)-x0(end))^2+(y0(1)-y0(end))^2);
d=sort(D*rand);
h=Parameters.Curliness(randi(size(Parameters.Curliness,2)))*D;

x1=x0(1);
y1=y0(1);
x2=x0(end);
y2=y0(end);

y31= (x1^2*y2 + x2^2*y2 - 2*y1*y2^2 + y1^2*y2 + y2^3 - 2*x1*x2*y2 + d*y1*(x1^2 - 2*x1*x2 + x2^2 + y1^2 - 2*y1*y2 + y2^2)^(1/2) - d*y2*(x1^2 - 2*x1*x2 + x2^2 + y1^2 - 2*y1*y2 + y2^2)^(1/2))/(x1^2 - 2*x1*x2 + x2^2 + y1^2 - 2*y1*y2 + y2^2);
x31=x1-(y1-y31)*(x1-x2)/(y1-y2);

y32=(x1^2*y2 + x2^2*y2 - 2*y1*y2^2 + y1^2*y2 + y2^3 - 2*x1*x2*y2 - d*y1*(x1^2 - 2*x1*x2 + x2^2 + y1^2 - 2*y1*y2 + y2^2)^(1/2) + d*y2*(x1^2 - 2*x1*x2 + x2^2 + y1^2 - 2*y1*y2 + y2^2)^(1/2))/(x1^2 - 2*x1*x2 + x2^2 + y1^2 - 2*y1*y2 + y2^2);
x32=x1-(y1-y32)*(x1-x2)/(y1-y2);

y3=y31;
x3=x31;
if (x1-x3)^2+(y1-y3)^2>(x1-x2)^2+(y1-y2)^2
    y3=y32;
    x3=x32;
end
x3=min([x3 M-11]);
y3=min([y3 N-11]);

m=-(x2-x1)/(y2-y1);

x41=(x3 + h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1);
y41=m*(x41-x3)+y3;
x42=(x3 - h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1);
y42=m*(x42-x3)+y3;

x_out=min([(x3 + h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1) M-11]);
y_out=min([m*(x41-x3)+y3 N-11]);

if      (x3 + h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1)>0 && m*(x41-x3)+y3>0
    x_out=min([(x3 + h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1) M-11]);
    y_out=min([m*(x41-x3)+y3 N-11]);
elseif  (x3 - h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1)>0 && m*(x42-x3)+y3>0
    x_out=min([(x3 - h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1) M-11]);
    y_out=min([m*(x42-x3)+y3 N-11]);
elseif (x3 + h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1)>0 && m*(x41-x3)+y3<0
    x_out=min([(x3 + h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1) M-11]);
    y_out=11;
elseif (x3 + h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1)<0 && m*(x41-x3)+y3>0
    x_out=11;
    y_out=min([m*(x41-x3)+y3 N-11]);
elseif (x3 - h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1)>0 && m*(x42-x3)+y3<0
    x_out=min([(x3 - h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1) M-11]);
    y_out=11;
elseif (x3 - h*(m^2 + 1)^(1/2) + m^2*x3)/(m^2 + 1)<0 && m*(x42-x3)+y3>0
    x_out=11;
    y_out=min([m*(x42-x3)+y3 N-11]);
end
x0=[x0(1) x0(1) x_out x0(end) x0(end)]';
y0=[y0(1) y0(1) y_out y0(end) y0(end)]';
end
function [x1,y1]=sortRndPnts(x0,y0);
XY=[x0 y0];
x_c=repmat(x0,1,size(x0,1));
y_c=repmat(y0,1,size(x0,1));
x_r=repmat(x0',size(x0,1),1);
y_r=repmat(y0',size(x0,1),1);

Diff=sqrt((x_c-x_r).^2+(y_c-y_r).^2);

[C,I]=max(Diff);
[r,c]=ind2sub(size(Diff),I(1));
x1(1)=x0(r);
y1(1)=y0(r);
x1(4)=x0(c);
y1(4)=y0(c);
x1=x1';
y1=y1';




end
function [Parameters]=parameter_Initialization(PhantImg,Parameters,ImagePath,size_Img)
if  Parameters.mode==1
    imshow(PhantImg)
elseif  Parameters.mode==2
    imshow(PhantImg)
    Parameters.generation_mode='On Fig1, select a set of points (at least 4).';
elseif Parameters.mode==3
    imshow(imread(ImagePath.Img_hair_occluded))
    Parameters.generation_mode='On Fig1, select a set of points (at least 4) along a hair shaft.';
end
if ~size(Parameters.Thickness,1)
    Parameters.Thickness=1+randi(10,1,8);
end
if ~size(Parameters.sigma,1)
    Parameters.sigma=1+randi(10,1,8);
end
if ~size(Parameters.Colors,1)
    Parameters.Colors=ListOfColors;
end
if ~size(Parameters.Curliness,1)
    Parameters.Curliness=randi(20,1,8);
end
end
function [mask,XX,YY,FlagcheckOutOfRangeSubscript]=hairMask(x,y,M,N)
n=4;
t = [ zeros(1, n-1) linspace(0,1,numel(x')-n+2) ones(1, n-1) ];  % knot vector
X = bspline_wdeboor(n,t,[x';y';zeros(1,size(x,1))],ones(1,size(x,1)));


alpha=1:size(X,2);
XX= fix(interp1(alpha,X(2,:),1:.05:size(X,2)));
YY= fix(interp1(alpha,X(1,:),1:.05:size(X,2)));

mask=zeros(M,N);
siz=[M N];
XXYY=[ YY'  XX'];
FlagcheckOutOfRangeSubscript=0;
for i = 1:2
    v = XXYY(:,i);
    if (any(v(:) < 1)) || (any(v(:) > siz(i)))
        FlagcheckOutOfRangeSubscript=1;
    end
end

if ~FlagcheckOutOfRangeSubscript
    IND=sub2ind([M N],YY,XX);
    mask(IND)=1;
    figure (1);hold on;plot(X(2,:),X(1,:),'-b','linewidth',2);
else
    disp('  ')
    disp('All the points should be inside the figure! ')
    disp('You insert some of them outside the eligible region!')
    disp('  ')
end

end


function mask2=HairDilation(XX,YY,M,N,mask,thickness)
mask2=ones(M,N);
delta=120;
for ii=10:length(YY)/2
    radiSE=round(ii/delta);
    if radiSE>thickness
        radiSE=thickness;
    end
    SE = strel('disk', radiSE, 8);
    mask2(YY(ii)-radiSE:YY(ii)+radiSE,XX(ii)-radiSE:XX(ii)+radiSE)=...
        imerode((mask( YY(ii)-radiSE:YY(ii)+radiSE, XX(ii)-radiSE:XX(ii)+radiSE)),SE);
    mask2(YY(end-ii)-radiSE:YY(end-ii)+radiSE,XX(end-ii)-radiSE:XX(end-ii)+radiSE)=...
        imerode((mask( YY(end-ii)-radiSE:YY(end-ii)+radiSE, XX(end-ii)-radiSE:XX(end-ii)+radiSE)),SE);
end
end
function img=ColorFulHair(mask,ImgRGB,clr,sigma)
h = fspecial('gaussian', 2*round(sigma+.49),sigma);
J=im2double(mask);
J=J./max(J(:));

blurred = imfilter(J,h,'replicate');
maskFin=repmat(blurred,[1 1 3]);
maskFin=1-maskFin;

maskFin2=maskFin;
maskFin2(:,:,1)=maskFin(:,:,1).*clr(1);
maskFin2(:,:,2)=maskFin(:,:,2).*clr(2);
maskFin2(:,:,3)=maskFin(:,:,3).*clr(3);
img=im2double(maskFin2)+im2double(ImgRGB).*(1-im2double(maskFin));%
end
function [C,u] = bspline_wdeboor(n,t,P,w,u)
% Evaluate explicit weighed B-spline at specified locations.
%
% Input arguments:
% n:
%    B-spline order (2 for linear, 3 for quadratic, etc.)
% t:
%    knot vector
% P:
%    control points, typically 2-by-m, 3-by-m or 4-by-m (for weights)
% w:
%    weight vector
% u (optional):
%    values where the B-spline is to be evaluated, or a positive
%    integer to set the number of points to automatically allocate
% Output arguments:
% C:
%    points of the B-spline curve

% Copyright 2010 Levente Hunyadi

w = transpose(w(:));
P = bsxfun(@times, P, w);
P = [P ; w];  % add weights to control points

if nargin >= 5
    [Y,u] = bspline_deboor(n,t,P,u);
else
    [Y,u] = bspline_deboor(n,t,P);
end

C = bsxfun(@rdivide, Y(1:end-1,:), Y(end,:));  % normalize and remove weights from computed points
end
function [C,U] = bspline_deboor(n,t,P,U)
% Evaluate explicit B-spline at specified locations.
%
% Input arguments:
% n:
%    B-spline order (2 for linear, 3 for quadratic, etc.)
% t:
%    knot vector
% P:
%    control points, typically 2-by-m, 3-by-m or 4-by-m (for weights)
% u (optional):
%    values where the B-spline is to be evaluated, or a positive
%    integer to set the number of points to automatically allocate
%
% Output arguments:
% C:
%    points of the B-spline curve
% Copyright 2010 Levente Hunyadi

validateattributes(n, {'numeric'}, {'positive','integer','scalar'});
d = n-1;  % B-spline polynomial degree (1 for linear, 2 for quadratic, etc.)
validateattributes(t, {'numeric'}, {'real','vector'});
assert(all( t(2:end)-t(1:end-1) >= 0 ), 'bspline:deboor:InvalidArgumentValue', ...
    'Knot vector values should be nondecreasing.');
validateattributes(P, {'numeric'}, {'real','2d'});
nctrl = numel(t)-(d+1);
assert(size(P,2) == nctrl, 'bspline:deboor:DimensionMismatch', ...
    'Invalid number of control points, %d given, %d required.', size(P,2), nctrl);
if nargin < 4
    U = linspace(t(d+1), t(end-d), 10*size(P,2));  % allocate points uniformly
elseif isscalar(U) && U > 1
    validateattributes(U, {'numeric'}, {'positive','integer','scalar'});
    U = linspace(t(d+1), t(end-d), U);  % allocate points uniformly
else
    validateattributes(U, {'numeric'}, {'real','vector'});
    assert(all( U >= t(d+1) & U <= t(end-d) ), 'bspline:deboor:InvalidArgumentValue', ...
        'Value outside permitted knot vector value range.');
end

m = size(P,1);  % dimension of control points
t = t(:).';     % knot sequence
U = U(:);
S = sum(bsxfun(@eq, U, t), 2);  % multiplicity of u in t (0 <= s <= d+1)
I = bspline_deboor_interval(U,t);

Pk = zeros(m,d+1,d+1);
a = zeros(d+1,d+1);

C = zeros(size(P,1), numel(U));
for j = 1 : numel(U)
    u = U(j);
    s = S(j);
    ix = I(j);
    Pk(:) = 0;
    a(:) = 0;
    
    % identify d+1 relevant control points
    Pk(:, (ix-d):(ix-s), 1) = P(:, (ix-d):(ix-s));
    h = d - s;
    
    if h > 0
        % de Boor recursion formula
        for r = 1 : h
            q = ix-1;
            for i = (q-d+r) : (q-s);
                a(i+1,r+1) = (u-t(i+1)) / (t(i+d-r+1+1)-t(i+1));
                Pk(:,i+1,r+1) = (1-a(i+1,r+1)) * Pk(:,i,r) + a(i+1,r+1) * Pk(:,i+1,r);
            end
        end
        C(:,j) = Pk(:,ix-s,d-s+1);  % extract value from triangular computation scheme
    elseif ix == numel(t)  % last control point is a special case
        C(:,j) = P(:,end);
    else
        C(:,j) = P(:,ix-d);
    end
end

    function ix = bspline_deboor_interval(u,t)
        % Index of knot in knot sequence not less than the value of u.
        % If knot has multiplicity greater than 1, the highest index is returned.
        
        i = bsxfun(@ge, u, t) & bsxfun(@lt, u, [t(2:end) 2*t(end)]);  % indicator of knot interval in which u is
        [row,col] = find(i);
        [row,ind] = sort(row);  %#ok<ASGLU> % restore original order of data points
        ix = col(ind);
    end
end
