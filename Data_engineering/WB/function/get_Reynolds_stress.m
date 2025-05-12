function [ Reynolds_stress ] = get_Reynolds_stress( uu, uv, uw, vu, vv, vw, wu, wv, ww )

% u -----------------------------------------------------------------------
if isempty(uu(:))
    disp('uu is empty, function is broken');
    return
end
if isempty(uv(:))
    uv = zeros(size(uu));
end
if isempty(uw(:))
    uw = zeros(size(uu));
end
% v -----------------------------------------------------------------------
if isempty(vu)
    vu = zeros(size(vv));
end
if isempty(vv(:))
   disp('vv is empty, function is broken');
   return
end
if isempty(vw(:))
    vw = zeros(size(vv));
end
% w -----------------------------------------------------------------------
if isempty(wu(:))
    wu = zeros(size(ww));
end
if isempty(wv(:))
    wv = zeros(size(ww));
end
if isempty(ww(:))
   disp('ww is empty, function is broken');
   return
end

vu = uv;
wu = uw;
wv = vw;

len = length(uu);
Reynolds_stress = cell(len, 1);
for i = 1:len
        Reynolds_stress{i} = [uu(i), uv(i), uw(i); vu(i), vv(i), vw(i); wu(i), wv(i), ww(i)];
end

end
