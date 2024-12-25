function [x,record_e] = lasso_ista(y,A,lambda,inter)

[~,n] = size(A);
t = 1;
L = norm(A,2)^2;

AtA = A'*A;
%AtA=A*A;
Aty = A'*y; 

x = zeros(n,1);
i=0;
record_e = zeros(inter,1);
% while true
for iter = 1 : inter
    df = 1/L*(AtA*x - Aty);
    x_last = x;
    x = x - df;
    x = sign(x).*max(abs(x) - (t/L)*lambda,0);
    i=i+1;
%     norm(x-x_last)
    record_e(iter) = norm(x-x_last);
    if norm(x-x_last) < 1e-6
        break;  
    end
end
