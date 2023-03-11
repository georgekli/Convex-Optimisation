function [pathx,pathl,pathv,rf,httas,iter]=dual_interior_point(A,b,c,x0,mi)
%Set tolerance for exit condition for residuals
ePrim=10^(-8);
eDual=10^(-8);
epsilon=10^(-8);
%Get problem dimentions and initialize
n=size(A,2);
p=size(A,1);
xval=x0;
lambda=-1./(-xval);
v=zeros(size(A,1),1);
%Set backtracking parameters
beta=0.5;
alpha=0.01;
iter=1;
%Set paths of output to initial values
pathx=xval;
pathl=lambda;
pathv=v;
%Set value m uset to calculate t in every iteration
mi=mi;
%Calculate htta
htta=-(-xval')*lambda;
httas=htta;
%Set t without the use of m for first iteration
t=n/htta;
while 1
    %Compute Dual Centering and Primal residuals
    httas=[httas htta];
    rDual=c+(-eye(n)')*lambda+A'*v;
    rCent=-diag(lambda)*(-xval)-1/t.*ones(n,1);
    rPrim=A*xval-b;
    r=[rDual ;rCent ;rPrim];
    if iter==1
        rf=sqrt(norm(rPrim)^2+norm(rDual)^2);
    end
    %Construct First Part of linear equation resolving in ?ypd
    firstPart=[ zeros(n,n) -eye(n) A';
                -diag(lambda)*(-eye(n)) -diag(-xval) zeros(n,p);
                A zeros(p,n) zeros(p,p)];
    secondPart=-r;
    %Solve the equation
    [sol]=linsolve(firstPart,secondPart);
    %Aquire Deltas for x,lambda and v
    xDelta=sol(1:n);
    lambdaDelta=sol(n+1:2*n);
    vDelta=sol(2*n+1:end);
    %Check ?>0 and backtrack var s from s=1
    s=1;
    while all(lambda+s*lambdaDelta<0)
        s=beta*s;
    end
    %Check x+s*?x>=0 and backtrack var s further
    x_new=xval+s*xDelta;
    while (all(x_new<0))
        s = beta * s; 
        x_new=xval+s*xDelta;
    end
    lambda_new=lambda+s*lambdaDelta;
    v_new=v+s*vDelta;
    %Compute residuals in y+=y+s*?y
    rDual_new=c+(-eye(n))*lambda_new+A'*v_new;
    rCent_new=-diag(lambda_new)*(-x_new)-1/t.*ones(n,1);
    rPrim_new=A*x_new-b;
    r_new=[rDual_new; rCent_new; rPrim_new];
    %Backtrack s until residuals satisfy condition
    while (norm(r_new)>(1-alpha*s)*norm(r))
        s = beta * s; 
        x_new=xval+s*xDelta;
        lambda_new=lambda+s*lambdaDelta;
        v_new=v+s*vDelta;
        %Recalculate new residuals for s
        rDual_new=c+(-eye(n))*lambda_new+A'*v_new;
        rCent_new=-diag(lambda_new)*(-x_new)-1/t.*ones(n,1);
        rPrim_new=A*x_new-b;
        r_new=[rDual_new; rCent_new; rPrim_new];
    end
    %Set x,lambda and v to new values now that s is settled
    xval=xval+s*xDelta;
    lambda=lambda+s*lambdaDelta;
    v=v+s*vDelta;
    %Add x,lambda,v to output path
    pathx=[pathx xval];
    pathl=[pathl lambda];
    pathv=[pathv v];
    %Compute new residuals in y+ now that s is settled
    rPrim=A*xval-b;
    rDual=c+(-eye(n))*lambda+A'*v;
    rTmp=sqrt(norm(rPrim)^2+norm(rDual)^2);
    rf=[rf rTmp];
    %Compute htta for new x and lambda
    htta=-(-xval')*lambda;
    %Check if residuals are close to zero 
    %fprintf("rPrim:%f\trDual:%f\thtta:%f\n",norm(rPrim),norm(rDual),htta);
    if norm(rPrim)<=ePrim && norm(rDual)<=eDual && htta<=epsilon
        %Exit condition
        break;
    end
    iter=iter+1;
    %Compute new value t+ for next iteration 
    t=mi*n/htta;
end
end