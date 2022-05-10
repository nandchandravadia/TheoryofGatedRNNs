%jacobian calculation


%% define symbols 

syms h1 h2; %state variables
syms g a; %parameters
syms J11 J12 J21 J22 W11 W12 W21 W22; %connectivity values

%% compute Jacobian for (n = 2)

jacobian([(-h1 + J11*tanh(g*h1) + J12*tanh(g*h2))/(1 + exp(-a*(W11*h1 + W12*h2))), 
    (-h2 + J21*tanh(g*h1) + J22*tanh(g*h2))/(1 + exp(-a*(W21*h1 + W22*h2)))],[h1,h2])