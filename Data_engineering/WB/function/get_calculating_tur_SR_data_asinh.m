function [ S_tensor, R_tensor ] = get_calculating_tur_SR_data_asinh( Ux, Vx, Wx, Uy, Vy, Wy, Uz, Vz, Wz )

Ux = reshape(Ux, [size(Ux,1)*size(Ux,2),1]);
Vx = reshape(Vx, [size(Vx,1)*size(Vx,2),1]);
Wx = reshape(Wx, [size(Wx,1)*size(Wx,2),1]);

Uy = reshape(Uy, [size(Uy,1)*size(Uy,2),1]);
Vy = reshape(Vy, [size(Vy,1)*size(Vy,2),1]);
Wy = reshape(Wy, [size(Wy,1)*size(Wy,2),1]);

Uz = reshape(Uz, [size(Uz,1)*size(Uz,2),1]);
Vz = reshape(Vz, [size(Vz,1)*size(Vz,2),1]);
Wz = reshape(Wz, [size(Wz,1)*size(Wz,2),1]);

len = length(Ux);
S_tensor = cell(len,1);
R_tensor = cell(len,1);

for i = 1:len
    [ S, R ] = get_strain_rotation_rate_tensor( Ux(i), Vx(i), Wx(i), Uy(i), Vy(i), Wy(i), Uz(i), Vz(i), Wz(i) );
    
    S_compressed = asinh(S);
    R_compressed = asinh(R);
        
    S_tensor{i} = S_compressed;
    R_tensor{i} = R_compressed;

    clear S R S_compressed R_compressed;
end

end



