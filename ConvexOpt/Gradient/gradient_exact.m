function [x,fun_val,fk]=gradient_exact(P,q,x0,epsilon);
% INPUT
% ======================
% P ....... the positive definite matrix
% q ....... a column vector associated with the linear part
% x0 ...... starting point of the method
% epsilon . tolerance parameter
% OUTPUT
% =======================
% x ....... an optimal solution (up to a tolerance) of min(1/2x^T A x2 b^T x)
% fun_val . the optimal function value up to a tolerance

x=x0;
iter=0;
grad=(P*x+q);
fk=[x(1) x(2) (1/2)*x'*P*x+q'*x];
while (norm(grad)>epsilon)
    iter=iter+1;
    t=norm(grad)^2/(grad'*P*grad);
    x=x-t*grad;
    grad=(P*x+q);
    fun_val=(1/2)*x'*P*x+q'*x;
    fk=[fk ;x(1) x(2) fun_val];
    fprintf('iter_number = %3d norm_grad = %2.6f fun_val = %2.6f \n',iter,norm(grad),fun_val);
end