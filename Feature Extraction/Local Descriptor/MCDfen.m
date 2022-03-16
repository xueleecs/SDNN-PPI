
% load set
function [group]=MCDfen(number)

    fast=number;
    N=length(fast);
    N1=floor(N/4);
    N2=floor(N/2);
    N3=3*N1;
    i=1;
    group{10*i-9}=fast(:,1:N1);
    group{10*i-8}=fast(:,N1+1:2*N1);
    group{10*i-7}=fast(:,2*N1+1:3*N1);
    group{10*i-6}=fast(:,3*N1+1:end);
    group{10*i-5}=fast(:,1:N2);
    group{10*i-4}=fast(:,N2:end);
    group{10*i-3}=fast(:,1:N3);
    group{10*i-2}=fast(:,N1:end);
    group{10*i-1}=fast(:,N1:N3);
    group{10*i}=fast;
end
    
    

   
    