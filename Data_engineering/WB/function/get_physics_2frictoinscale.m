function [X, Y, Z, U, V, W] = get_physics_2frictoinscale(X, Y, Z, U, V, W, scle_Ru, scle_Rl)

X = X*scle_Rl;
Y = Y*scle_Rl;
Z = Z*scle_Rl;
U = U*scle_Ru;
V = V*scle_Ru;
W = W*scle_Ru;

end