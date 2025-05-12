function [ x_data_point, y_data_point ] = set_ML_triangle_point_R2023( x_data, y_data, target_points, step_h);

% step_h = 1;
% target_points = 3000;


x_rest = [ 0.1427, 0.1427, 12,     12,     28,     28,     39.8573, 39.8573];
y_rest = [ 2.9596, 0.0311, 0.0311, step_h, step_h, 0.0311, 0.0311,  2.9596];

pgon = polyshape({x_rest}, {y_rest});
% plot(pgon);


tr = triangulation(pgon);
tnodes = tr.Points';
telements = tr.ConnectivityList';


model = createpde;
geometryFromMesh(model,tnodes,telements);
% pdegplot(model,"VertexLabels","on","EdgeLabels","on")


Hmax_values = linspace(0.6, 0.8, 25);
mesh_point_count = zeros(length(Hmax_values), 1);

for i = 1:length(Hmax_values)
    mesh = generateMesh(model, 'Hmax', Hmax_values(i), 'GeometricOrder', 'quadratic', Hgrad=1.09, Hedge={[1], 0.15, [5], 0.19, [7], 0.18, [4], 0.25, [8], 0.25});
    mesh_point_count(i) = size(mesh.Nodes, 2);
end

[~, idx] = min(abs(mesh_point_count - target_points));
optimal_Hmax = Hmax_values(idx);


mesh = generateMesh(model, 'Hmax', optimal_Hmax,  'GeometricOrder', 'quadratic', Hgrad=1.09, Hedge={[1], 0.15, [5], 0.19, [7], 0.18, [4], 0.25, [8], 0.25});
% pdeplot(mesh);


% title('Mesh with Refined Edges on the Waveform');
% axis equal;
x_data_point = mesh.Nodes(1,:);
y_data_point = mesh.Nodes(2,:);
x_data_point = x_data_point';
y_data_point = y_data_point';

end