function [xk,fun_vals]=newton_primal_dual(A,b,f,g,h,x0,alpha,beta,epsilon)
% Newton's method for primal dual solving with backtracking
% INPUT
%=======================================
% A ......... Ax=b contraint
% b ......... Ax=b contraint
% f ......... objective function
% g ......... gradient of the objective function
% h ......... hessian of the objective function
% x0......... initial point
% alpha ..... tolerance parameter for the stepsize selection strategy
% beta ...... the proportion in which the stepsize is multiplied
%             at each backtracking step (0<beta<1)
% epsilon ... tolerance parameter for stopping rule
% OUTPUT
%=======================================
% xk......... solution steps
% fun_vals... steps to optimal value

xval=x0;
vval=zeros(size(A,1),1);
xCell=num2cell(xval);
vCell=num2cell(vval);
sCell=[xCell;vCell];
fun_val=double(f(xCell{:}));
gval=double(g(xCell{:}));
hval=double(h(xCell{:}));
iter=0;
fun_vals=(fun_val);
xk=xCell;
x=sym('x',[size(A,2) 1]);
v=sym('v',[size(A,1) 1]);
rd=symfun(g+A'*v,[x ;v]);
rp=symfun(A*x-b,[x ;v]);
r=[rd ; rp];
while (1)
    iter=iter+1;
    firstPart=[hval A';A zeros(size(A,1))];
    secondPart=[-gval;(-A*xval+b)];
    [sol]=linsolve(firstPart,secondPart);
    xDelta=sol(1:size(A,2));
    w=sol(size(A,2)+1:end);
    vDelta=w-vval;
    t=1;
    xCellTmp=num2cell(xval+t*xDelta);
    vCellTmp=num2cell(vval+t*vDelta);
    sCellTmp=[xCellTmp;vCellTmp];
    while(norm(double(r(sCellTmp{:}))) > (1-alpha*t)*norm(double(r(sCell{:}))))
        %fprintf('\tBackTracking...\n'); 
        t=beta*t;
        xCellTmp=num2cell(xval+t*xDelta);
        vCellTmp=num2cell(vval+t*vDelta);
        sCellTmp=[xCellTmp;vCellTmp];
    end
    if t==1
        fprintf('Feasible!!!!\n');
    end
    xval=xval+t*xDelta;
    vval=vval+t*vDelta;
    xCell=num2cell(xval);
    vCell=num2cell(vval);
    sCell=[xCell;vCell];
    fun_val=double(f(xCell{:}));
    fprintf('iter= %2d f(x)=%10.10f',iter,fun_val)
    fun_vals=[fun_vals fun_val];
    xk=[xk xCell];
    gval=double(g(xCell{:}));
    hval=double(h(xCell{:}));
    fprintf(' norm= %f\n',norm(double(r(sCell{:}))))
    if norm(double(r(sCell{:})))<=epsilon
        return
    end
end