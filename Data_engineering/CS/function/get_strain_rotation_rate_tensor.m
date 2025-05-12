function [ S, R ] = get_strain_rotation_rate_tensor( Ux, Vx, Wx, Uy, Vy, Wy, Uz, Vz, Wz )

S = 0.5.*[Ux+Ux Uy+Vx Uz+Wx; Vx+Uy Vy+Vy Vz+Wy; Wx+Uz Wy+Vz Wz+Wz];
R = 0.5.*[Ux-Ux Uy-Vx Uz-Wx; Vx-Uy Vy-Vy Vz-Wy; Wx-Uz Wy-Vz Wz-Wz];

end

