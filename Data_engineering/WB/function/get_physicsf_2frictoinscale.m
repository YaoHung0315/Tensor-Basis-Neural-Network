function [uu, vv, ww, uv, uw, vw] = get_physicsf_2frictoinscale(uu, vv, ww, uv, uw, vw, scle_Ru)

uu = uu*scle_Ru^2;
vv = vv*scle_Ru^2;
ww = ww*scle_Ru^2;
uv = uv*scle_Ru^2;
uw = uw*scle_Ru^2;
vw = vw*scle_Ru^2;

end