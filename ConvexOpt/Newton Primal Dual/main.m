clc;
clear all;
close all;
%Part1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=10;
n=20;
A=rand(p,n);
for i=1:n
    x(i)=rand();
end
x=x';
b=A*x;
%Part2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cvx_begin
    variable x(n)
    minimize(-sum(log(x)));
    subject to
        A*x==b;
cvx_end
optP=cvx_optval
xOpt=x;
cvx_begin
    variable x(n)
    
    subject to
        A*x==b;
        x>=0;
cvx_end
xStart=x;
x=sym('x',[n 1]);
f=symfun(-sum(log(x)),x);
g=gradient(f);
h=hessian(f);
%h=diff(f,2);
tolerance=10^(-3);
[x1,fun_val1]=newton_affine_constraint(A,f,g,h,xStart,0.3,0.25,tolerance);
figure()
x1=cell2mat(x1);
scatter(1:n,xOpt,'*');
legend('cvx_x');
hold on;
for i=1:size(x1,2)
    scatter(1:n,x1(:,i));
    hold on;
end
hold off;
%Part3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nStarting primal dual\n');
xStart=ones(size(A,2),1);
[x2,fun_val2]=newton_primal_dual(A,b,f,g,h,xStart,0.25,0.5,tolerance);
figure()
x2=cell2mat(x2);
scatter(1:n,xOpt,'*');
legend('cvx_x');
hold on;
for i=1:size(x2,2)
    scatter(1:n,x2(:,i));
    hold on;
end
hold off;
%Part4%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cvx_begin
    variable v(p)
    maximize(-b'*v+sum(log(A'*v)+n));
cvx_end
optx=1./(A'*v);
xCell=num2cell(optx);
opt_val=double(f(xCell{:}))
%Part5%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(p==1 && n==2)
    figure()
    fsurf(f);
    hold on;
    %x1=cell2mat(x1);
    pause(2);
    for i=1:size(x1,2)
        plot3(x1(1,i),x1(2,i),fun_val1(i), '.', 'MarkerSize', 30);
        pause(1);
    end
    pause(2);
    figure()
    fsurf(f);
    hold on;
    %x2=cell2mat(x2);
    pause(2);
    for i=1:size(x2,2)
        plot3(x2(1,i),x2(2,i),fun_val2(i), '.', 'MarkerSize', 30);
        pause(1);
    end
end
%Expand%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure()
diff1=abs(opt_val-fun_val1);
diff2=abs(opt_val-fun_val2);
plot(1:size(fun_val1,2),diff1);
hold on;
plot(1:size(fun_val2,2),diff2);
legend('Newton','Primal-Dual');
hold off;