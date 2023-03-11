%% 
clc;
clear all;
close all;
%Consruct%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=50;
n=100;
A=rand(p,n);
for i=1:n
    x(i)=rand();
    s(i)=rand(); %used to compute cost function later
end
x=x';
s=s';
b=A*x;
z=normrnd(0,1,[p 1]);
c=A'*z+s;
%% 
%Part1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cvx_begin
    variable x(n)
    minimize(c'*x);
    subject to
        A*x==b;
        x>=0;
cvx_end
optP=cvx_optval;
xOpt=x;
%% 
%Part2a%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cvx_begin
    variable x(n)
    subject to
        A*x==b;
        x>=0;
cvx_end
xFeasible=x;
[xk,rl,o_iter,i_iter,cumIters]=interior_point(A,b,c,xFeasible);
xIntPointOptA=xk(:,end);
sum((xIntPointOptA-xOpt))
%% 
%Part2b%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate Feasible Point
xFeas=find_feasible_point(A,b,c);
[xkSec,rlSec,o_iterSec,i_iterSec,cumItersSec]=interior_point(A,b,c,xFeas);
xIntPointOptB=xkSec(:,end);
sum((xIntPointOptB-xOpt))
%% 
%Part3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xInfeasible=ones(n,1);
mi=2;
[xPath,lPath,vPath,rf,httas,pd_iter]=dual_interior_point(A,b,c,xFeasible,mi);
%[xPath,lPath,vPath,rf,httas,pd_iter]=dual_interior_point(A,b,c,xInfeasible,mi);
sum((xPath(:,end)-xOpt))
%%
%Algorithm Exploration%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
epsilon=10^(-6);
figure()
bar(1:n,xOpt);
title("Optimal vector x");
ylabel('x(i)') 
xlabel('i') 
fprintf("0s in optimal vector:%d for p=%d\n",sum(xOpt<epsilon),p);
if p==1 && n==2
    figure()
    title("x^k for the 2 IPM algorithms");
    xlabel=("x1");
    ylabel=("x2");
    plot(xOpt(1),xOpt(2),"*",'MarkerSize',10);
    hold on;
    plot(xk(1,:),xk(2,:),'.');
    hold on;
    plot(xPath(1,:),xPath(2,:),'x');
    legend("y^*","Newton IPM","Primal-Dual IPM");
    xlim([-0.5 max(max(xk(1,:)),max(xPath(1,:)))+0.5]);
    ylim([-0.5 max(max(xk(2,:)),max(xPath(2,:)))+0.5]);
end
if p~=1 && n~=2
    figure()
    semilogy(1:cumIters,rl(2:end));
    xlabel("Iterations");
    ylabel("1/t");
    title("Newton IPM");
    figure()
    plot(1:pd_iter,rf(2:end));
    xlabel("Iterations");
    ylabel("rFeas");
    title("Primal-Dual IPM");
    figure()
    plot(1:pd_iter,httas(2:end));
    xlabel("Iterations");
    ylabel("htta");
    title("Primal-Dual IPM");
end
