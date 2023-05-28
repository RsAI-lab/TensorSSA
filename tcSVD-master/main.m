close all;
clear all;
clc;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%test incremental t-SVD Tensor
n=30;
A=rand(200,400,n);
B=rand(200,10,n);
Y=[A B];

% AA=reshape(1:12,[2,2,3]);
% r=1;
% [U,S,V] = tSVD(AA,r);
% C = tProdact(U,S);
% R=tProdact(C,tTranspose(V));

[uA,sA,vA]=tSVD(A);
[unew,snew,vnew]=incrementalTSVD(uA,sA,vA,B,Y);

error1=A-tProdact(uA,tProdact(sA,tTranspose(vA)));
normerror1=norm(error1(:))

errorIncremental=Y-tProdact(unew,tProdact(snew,tTranspose(vnew)));
normerrorIncremental=norm(errorIncremental(:))


[uY,sY,vY]=tSVD(Y);
errorDirect=Y-tProdact(uY,tProdact(sY,tTranspose(vY)));
normerrorDirect=norm(errorDirect(:))


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%test incremental SVD matrix

Amat=rand(3,5);Bmat=rand(3,10); Ymat=[Amat Bmat];
[uA,sA,vA]=svd(Amat,'econ');
[ui,si,vi]=incrementalSVD(uA,sA,vA,Bmat,false);
[uY,sY,vY]=svd(Ymat);
normY=norm(Ymat-ui*si*vi')
