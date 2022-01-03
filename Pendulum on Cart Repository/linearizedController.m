load('linearizedUp.mat')
% load('linearizedDown.mat')


% A1 = linearizedDown.A;
% B1 = linearizedDown.B;

A2 = linearizedUp.A;
B2 = linearizedUp.B;
% C2 = [1 1 1 1];
% D2 = 0;


% Q1 = ctrb(A1,B1);
% rank(Q1)
% 
% Q2 = ctrb(A2,B2);
% rank(Q2)


eigs = [-3; -4; -3.5; -4.5];
% K1 = place(A1, B1, eigs);
K2 = place(A2, B2, eigs);


% k2i = 1;


% A2_tilde = [[A2-B2*K2; -C2+D2*K2] [-B2*k2i; D2*k2i]];
% eig(A2_tilde);