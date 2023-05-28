%% A code for TSVD
function [U,S,V] = tSVD(A,r)
    [n1,n2,n3]=size(A);
    D=fft(A,[],3);
    
    if nargin<=1
        U=zeros([n1,min([n1,n2]),n3]);S=zeros([min([n1,n2]),min([n1,n2]),n3]);V=zeros([n2,min([n1,n2]),n3]);
        for i=1:n3
            [u,s,v]=svd(double(D(:,:,i)),'econ');
            U(:,:,i)=u;
            S(:,:,i)=s;
            V(:,:,i)=v;
        end
    else
        U=zeros([n1,r,n3]);S=zeros([r,r,n3]);V=zeros([n2,r,n3]);
        for i=1:n3
            %svds函数分解
            [u,s,v]=svds(double(D(:,:,i)),r);
            
%             %直接计算求解
%             DD=double(D(:,:,i))*double(D(:,:,i))';
%             [u,autoval]=eigs(DD,r);
%             s=sqrt(autoval);
%             v=(double(D(:,:,i))')*u/s;
            
            %Fast Randomized SVD分解，但精度较低
            %[u,s,v]=rsvd(double(D(:,:,i)),r);

            U(:,:,i)=u;
            S(:,:,i)=s;
            V(:,:,i)=v;
        end
    end

    U=ifft(U,[],3);
    S=ifft(S,[],3);
    V=ifft(V,[],3);
end