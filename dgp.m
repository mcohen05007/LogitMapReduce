function Data = dgp()
N = 1000;                            % Number of HH
J = 3;                              % Number of Payment types
a = 49.5;		
b = 49.5;
T = round(a+(b-a)*rand(1,N));       % Number of transactions per household
theta1 = [-1];                % Marginal utility of expenditure for each payment type
int = [.5 .25 0];
P = rand(sum(T),J);               % Price
% Compute Probabilities
eu = exp(log(P)*theta1 + ones(sum(T),1)*int);
Pr = eu./(sum(eu,2)*ones(1,J));
% Draw Choices
Y = mnrnd(1,Pr);                    % Expressed as indicator
[choice,j] = find(Y');              % Expressed as cardinal
p = log(P)';
X = [repmat([eye(J-1);zeros(1,J-1)],sum(T),1) p(:)];
Xt=reshape(X',size(X,2)*J,sum(T))';
names = {'int11','int12','Price1','int21','int22','Price2','int31','int32','Price3','y1','y2','y3'};
Data = array2table([Xt Y],'VariableNames',names);
end