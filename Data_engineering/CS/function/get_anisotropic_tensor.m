function [ anisotropic_tensor ] = get_anisotropic_tensor( Reynolds_stress )

len = length(Reynolds_stress);
anisotropic_tensor = cell(len, 1);
indentity_matrix = eye(3);
for i = 1:len

    temp_matrix = Reynolds_stress{i};
    temp_principle_elements = temp_matrix(1,1) + temp_matrix(2,2) + temp_matrix(3,3);
    temp_anisotropic_tensor = temp_matrix - 1./3.*temp_principle_elements.*indentity_matrix;
    anisotropic_tensor{i} = temp_anisotropic_tensor;

    clear temp_matrix;

end

end

