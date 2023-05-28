%==========================================================================
% H. Fu, et al, "Tensor Singular Spectral Analysis for 3D feature extraction in hyperspectral images"
% TensorSSA on IP, PU, MG
%==========================================================================
clear all;close all;clc
addpath(genpath('.\tcSVD-master'));
addpath(genpath('.\HSIdata'));
addpath(genpath('.\libsvm-3.18'));

%% load dataset
load('indian_pines_gt'); img_gt=indian_pines_gt; 
load('Indian_pines_corrected');img=indian_pines_corrected;
[W,H,B]=size(img);

% load('PaviaU_gt'); img_gt=paviaU_gt;
% load('PaviaU');img=paviaU;
% [W,H,B]=size(img);

% load('MUUFL_Gulfport_gt'); img_gt=MUUFL_Gulfport_gt;
% load('MUUFL_Gulfport');img=MUUFL_Gulfport;
% [W,H,B]=size(img);

%% parameters (Can be changed as required)
%best parameters for IP
u=5; w=2*u+1; w2=w*w;  %research region
L=49;                  %embedding L

% %best parameters for PU
% u=2; w=2*u+1; w2=w*w;  %research region
% L=9;                  %embedding L

% %best parameters for MG
% u=3; w=2*u+1; w2=w*w;  %research region
% L=25;                  %embedding L

%% TensorSSA
tic;
indian_pines = padarray(img,[u,u],'symmetric','both');
Id=zeros(L,W*H);
Fea_cube=zeros(L,W*H,B);
%adaptive embedding
k=0;
for i=1:W
    for j=1:H
        i1=i+u;j1=j+u;k=k+1;
        testcube=indian_pines(i1-u:i1+u,j1-u:j1+u,:);
        m=reshape(testcube,[w2,B]);    
        %NED
        center=m((w2+1)/2,:);NED=zeros(1,w2);
        for ii=1:w2
            NED(:,ii)=sqrt(sum(power((m(ii,:)/norm(m(ii,:))-center/norm(center)),2)));%NED
        end
        [~,ind]=sort(NED);
        index=ind(1:L);
        Id(:,k)=index;
        Fea_cube(:,k,:)=m(index,:);
    end
end
%T-SVD decomposition
[U,S,V] = tSVD(Fea_cube,1); %rank=1,i.e.,Low-rank representation
clear Fea_cube;
C = tProdact(U,S);
VT=tTranspose(V);
Feacube_proc=tProdact(C,VT);
%Reprojection
New_pad_img=zeros(W+w-1,H+w-1,B);
repeat=zeros(W+w-1,H+w-1);
kk=0;
for i=1:W
    for j=1:H
        kk=kk+1;
        rec_m=zeros(w2,B);
        rec_m(Id(:,kk),:)=Feacube_proc(:,kk,:);
        dd=reshape(rec_m,[w,w,B]);
        
        rec_col=zeros(w2,1);
        rec_col(Id(:,kk))=1;
        
        i1=i+u;j1=j+u;
        New_pad_img(i1-u:i1+u,j1-u:j1+u,:)=New_pad_img(i1-u:i1+u,j1-u:j1+u,:)+dd;
        repeat(i1-u:i1+u,j1-u:j1+u)=repeat(i1-u:i1+u,j1-u:j1+u)+reshape(rec_col,w,w);
    end
end
New_pad_img=New_pad_img./repeat;
img_tensorSSA=New_pad_img(u+1:W+u,u+1:H+u,:);
toc;
clear Feacube_proc;

%% SVM based classification
Labels=reshape(img_gt,W*H,1);
Vectors=reshape(img_tensorSSA,W*H,B);  
class_num=max(max(img_gt))-min(min(img_gt));
trainVectors=[];trainLabels=[];train_index=[];
testVectors=[];testLabels=[];test_index=[];
rng('default');
Sam=0.02;              % Training sample ratio:0.02 for IP, 0.01 for PU and MG
for k=1:1:class_num
    index=find(Labels==k);           
    perclass_num=length(index);      
    Vectors_perclass=Vectors(index,:);   
    c=randperm(perclass_num);                
    select_train=Vectors_perclass(c(1:ceil(perclass_num*Sam)),:);  
    train_index_k=index(c(1:ceil(perclass_num*Sam)));
    train_index=[train_index;train_index_k];
    select_test=Vectors_perclass(c(ceil(perclass_num*Sam)+1:perclass_num),:);
    test_index_k=index(c(ceil(perclass_num*Sam)+1:perclass_num));
    test_index=[test_index;test_index_k];
    trainVectors=[trainVectors;select_train];                    
    trainLabels=[trainLabels;repmat(k,ceil(perclass_num*Sam),1)];
    testVectors=[testVectors;select_test];                      
    testLabels=[testLabels;repmat(k,perclass_num-ceil(perclass_num*Sam),1)];
end
[trainVectors,M,min] = scale_func(trainVectors);  %M、m分别为列的最大值和最小值
[testVectors ] = scale_func(testVectors,M,min);  
[Vectors ] = scale_func(Vectors,M,min);  
Ccv=1000; Gcv=0.1250;
%SVM training
cmd=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv); 
models=svmtrain(trainLabels,trainVectors,cmd);
%SVM test
testLabel_est= svmpredict(testLabels,testVectors, models);

GroudTest=testLabels;
SVMResultTest=testLabel_est;
[OA,AA,kappa,CA]=confusion(GroudTest,SVMResultTest);
ACC=[CA*100;OA*100;AA*100;kappa*100];   %classification accuracy
ACC

SVMresult = svmpredict(Labels,Vectors,models); 
SVMresult = reshape(SVMresult,W,H);
SVMmap = label2color(SVMresult,'india');  %for IP
% SVMmap = label2color(SVMresult,'uni');  %for PU
% SVMmap = label2color(SVMresult,'muufl'); %for MG
figure,imshow(SVMmap);

