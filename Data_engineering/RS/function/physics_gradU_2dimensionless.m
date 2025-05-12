function [Ux, Uy, Uz, Vx, Vy, Vz, Wx, Wy, Wz] = physics_gradU_2dimensionless(Ux, Uy, Uz, Vx, Vy, Vz, Wx, Wy, Wz, scle_L, scle_U)
Ratio = scle_L/scle_U;
Ux = Ux.*Ratio;
Uy = Uy.*Ratio;
Uz = Uz.*Ratio;
Vx = Vx.*Ratio;
Vy = Vy.*Ratio;
Vz = Vz.*Ratio;
Wx = Wx.*Ratio;
Wy = Wy.*Ratio;
Wz = Wz.*Ratio;
end
