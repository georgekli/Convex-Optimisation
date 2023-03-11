clc;
clear all;
close all;

%Part B%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=2;
A=rand(n);
[U,S,V]=svd(A);
lambda_min=0.1;
lambda_max=1;
z=lambda_min+(lambda_max-lambda_min)*rand(n-2,1);
eig_P=[lambda_min;lambda_max;z];
lambda=diag(eig_P);
P=U*lambda*(U');
q=rand(n,1);
tolerance=0.01;
x0=ones(n,1).*randn(1,1);
fun_val_x0=(1/2)*x0'*P*x0+q'*x0;
fprintf('Exact Begins\n');
[x,funval,fk]=gradient_exact(P,q,x0,tolerance);
x=sym('x',[2 1]);
f(x)=(1/2)*x'*P*x+q'*x;
figure();
fcontour(f,'LevelList',fk(:,3));
hold on;
plot(fk(:,1),fk(:,2));
metric=log(fk(:,3)'-funval);
k=1:1:size(fk(:,3),1);
figure()
plot(k,metric);
max_iter=lambda_max/lambda_min*log((fun_val_x0-funval)/tolerance);

fprintf('Backtracking Begins\n');
[x,fun_val,fk]=gradient_backtracking(P,q,x0,1,0.3,0.3,tolerance);
figure();
fcontour(f,'LevelList',fk(:,3));
hold on;
plot(fk(:,1),fk(:,2));
metric=log(fk(:,3)'-funval);
k=1:1:size(fk(:,3),1);
figure()
plot(k,metric);
max_iter=lambda_max/lambda_min*log((fun_val_x0-fun_val)/tolerance)
%Part C1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
m = 200; n = 50;
A = randn(m,n); b = abs(randn(m,1));
c = randn(n,1);
cvx_begin
    variable x(n,1)
    minimize(c'*x-sum(log(b-A*x)));
cvx_end
cvx_optval
x
if(n==2)
    X1=(x(1)-0.5):0.01:(x(1)+0.5);
    X2=(x(2)-0.5):0.01:(x(2)+0.5);
    X1=X1';
    X2=X2';
    f=zeros(size(X1,2),size(X1,2));
    for(i=1:size(X1,1))
        for(j=1:size(X1,1))
            for(k=1:1:m)
                if(b-A*[X1(i) X2(j)]'>0)
                    f(i,j)=c'*[X1(i) X2(j)]'-sum(log(b-A*[X1(i) X2(j)]'));
                else
                    f(i,j)=10^3;
                end
            end
        end
    end
    figure()
    mesh(X1,X2,f);
    hold on;
    contour(X1,X2,f,cvx_optval-0.2:0.1:cvx_optval+5);
    zlim([-100 100])
    figure()
    contour(X1,X2,f,cvx_optval-0.2:0.1:cvx_optval+5);
    hold on;
    plot(x(1),x(2));
end
%Part C2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=sym('x',[n 1]);
x0=zeros(n,1);
f=symfun(c'*x-sum(log(b-A*x)),x);
g=gradient(f);
h=hessian(f);
[x,fun_valGrad,fun_valsGrad]=gradient_method_backtracking(b,A,f,g,x0,1,0.3,0.5,0.01);
%Part C3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x,fun_valNew,fun_valsNew]=pure_newton(f,g,h,x0,0.01);
%Results%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure()
k1=1:1:size(fun_valsGrad,2);
k2=1:1:size(fun_valsNew,2);
semilogy(k1,fun_valsGrad-fun_valGrad);
hold on;
semilogy(k2,fun_valsNew-fun_valNew);
legend('Gradient','Newton');
ylabel('f(x_k)-p*');
xlabel('k')

