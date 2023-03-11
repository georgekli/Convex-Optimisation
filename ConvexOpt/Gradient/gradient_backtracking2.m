function [x,fun_val,fun_vals]=gradient_backtracking2(c,b,A,x0,s,alpha,beta,epsilon)
% Gradient method with backtracking stepsize rule
%
% INPUT
%=======================================
% c ......... vector c of equation
% b ......... vector b of equation
% A ......... matrix A of equation
% x0......... initial point
% s ......... initial choice of stepsize
% alpha ..... tolerance parameter for the stepsize selection
% beta ...... the constant in which the stepsize is multiplied 
%             at each backtracking step (0<beta<1)
% epsilon ... tolerance parameter for stopping rule
% OUTPUT
%=======================================
% x ......... optimal solution (up to a tolerance) 
%             of min f(x)
% fun_val ... optimal function value
x=x0;
tmp=zeros(size(A,2),1);
for i=1:size(A,1)
    tmp=tmp+(1/(b(i)-A(i,:)*x)*A(i,:)');
end
grad=c+tmp;
fun_val=c'*x-sum(log(b-A*x));
fun_vals=fun_val;
iter=0;
while (norm(grad)>epsilon)
    iter=iter+1;
    t=s;
    while(sum(b-A*(x-t*grad)<=0)>0)
        fprintf('\tGetting in domf\n');        
        t=beta*t;
    end
    while (fun_val-(c'*(x-t*grad)-sum(log(b-A*(x-t*grad))) )<alpha*t*norm(grad)^2)
        fprintf('\tbacktracking with t=%f...\n',t);
        t=beta*t;
    end
    x=x-t*grad;
    fun_val=c'*x-sum(log(b-A*x));
    tmp=zeros(size(A,2),1);
    for i=1:size(A,1)
        tmp=tmp+(1/(b(i)-A(i,:)*x)*A(i,:)');
    end
    grad=c+tmp;
    fprintf('iter_number = %3d norm_grad = %2.6f fun_val = %2.6f \n',iter,norm(grad),fun_val);
    fun_vals=[fun_vals fun_val];
end