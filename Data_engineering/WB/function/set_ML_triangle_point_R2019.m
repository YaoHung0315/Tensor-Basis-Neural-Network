function [ x_data_point, y_data_point ] = set_ML_triangle_point_R2019( x_data, y_data, target_points);

x_rest = [ x_data(1,end), x_data(:,1)', x_data(end,end)];
y_rest = [ y_data(1,end), y_data(:,1)', y_data(end,end)];

pgon = polyshape({x_rest}, {y_rest});
% plot(pgon);


tr = triangulation(pgon);
tnodes = tr.Points';
telements = tr.ConnectivityList';


model = createpde;
geometryFromMesh(model,tnodes,telements);
% pdegplot(model,"VertexLabels","on","EdgeLabels","on")

Hmax_values = linspace(0.05, 0.2, 50);
mesh_point_count = zeros(length(Hmax_values), 1);

for i = 1:length(Hmax_values)
    mesh = generateMesh(model, 'Hmax', Hmax_values(i), 'GeometricOrder', 'quadratic');
    mesh_point_count(i) = size(mesh.Nodes, 2);
end

[~, idx] = min(abs(mesh_point_count - target_points));
optimal_Hmax = Hmax_values(idx);

mesh = generateMesh(model, 'Hmax', optimal_Hmax, 'GeometricOrder', 'quadratic');
% pdeplot(mesh);
% title('Mesh with Refined Edges on the Waveform');
% axis equal;
x_data_point = mesh.Nodes(1,:);
y_data_point = mesh.Nodes(2,:);

x_data_point = x_data_point';
y_data_point = y_data_point';

end