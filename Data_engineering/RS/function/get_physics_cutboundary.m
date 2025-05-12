function [uu, vv, ww, uv, uw, vw] = get_physics_cutboundary(uu, vv, ww, uv, uw, vw)

uu = uu(2:end-1,2:end-1);
vv = vv(2:end-1,2:end-1);
ww = ww(2:end-1,2:end-1);
uv = uv(2:end-1,2:end-1);
uw = uw(2:end-1,2:end-1);
vw = vw(2:end-1,2:end-1);

end