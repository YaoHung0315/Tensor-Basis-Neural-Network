function [ T_matrix ] = get_polymatrix_2D( S, R )

T_1 = S;
T_2 = S*R - R*S;
T_3 = S*S - 1./3.*eye(size(S))*trace(S*S);
T_4 = R*R - 1./3.*eye(size(R))*trace(R*R);
T_5 = R*S*S - S*S*R;

T_6 = R*R*S + S*R*R - 2./3.*eye(size(S)).*trace(S*R*R);
T_7 = R*S*R*R - R*R*S*R;
T_8 = S*R*S*S - S*S*R*S;
T_9 = R*R*S*S + S*S*R*R - 2./3.*eye(size(S)).*trace(S*S*R*R);
T_10 = R*S*S*R*R - R*R*S*S*R;

T_matrix = {T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10};

end

