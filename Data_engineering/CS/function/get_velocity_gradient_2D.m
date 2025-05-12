function [Ux, Vx, Wx, Uy, Vy, Wy, Uz, Vz, Wz] = get_velocity_gradient_2D( X, Y, Z, U, V, W)

% The mean flow velocity can be 1 or 2 dimensions.
Ux = zeros(size(U));
Uy = zeros(size(U));
Uz = zeros(size(U));
Vx = zeros(size(U));
Vy = zeros(size(U));
Vz = zeros(size(U));
Wx = zeros(size(U));
Wy = zeros(size(U));
Wz = zeros(size(U));

% Mean flow 1D
sy = size(U,2);
if sy==1
    disp('sy==1');
    if isempty(X) && ~isempty(Y) && isempty(Z)
        if ~isempty(U)
            Uy = zeros(size(U));
            Uy(2:end-1) = (U(3:end)-U(1:end-2))./(Y(3:end)-Y(1:end-2));
        else
            disp('No mean U velocity in reading data ONE-dimensional case');
            return
        end
        %---remove the ghost points (central difference)---1D----------------------
        Ux = Ux(2:end-1, 1);
        Vx = Vx(2:end-1, 1);
        Wx = Wx(2:end-1, 1);
        
        Uy = Uy(2:end-1, 1);
        Vy = Vy(2:end-1, 1);
        Wy = Wy(2:end-1, 1);
        
        Uz = Uz(2:end-1, 1);
        Vz = Vz(2:end-1, 1);
        Wz = Wz(2:end-1, 1);
    end
end

% Mean flow 2D
if sy>=2
    disp('sy==2');
%     ~isempty(U) && ~isempty(V)
%     ~isempty(X) && ~isempty(Y) && ~isempty(Z)
    if 1;~isempty(X) && ~isempty(Y) && ~isempty(Z);
        if ~isempty(U) && ~isempty(V)
            % plot(slice_x(1:end-2,2:end-1),slice_y(1:end-2,2:end-1),'-r.') % xl
            % plot(slice_x(3:end,2:end-1),slice_y(3:end,2:end-1),'-b.') % xr
            % plot(slice_x(2:end-1,3:end),slice_y(2:end-1,3:end),'-y.') % yt
            % plot(slice_x(2:end-1,1:end-2),slice_y(2:end-1,1:end-2),'-c.') % yb
            Ux = zeros(size(U));
            Uy = zeros(size(U));
            Uxi   = U(3:end,2:end-1)-U(1:end-2,2:end-1); % difference of U in xi-direction
            Uyi   = U(2:end-1,3:end)-U(2:end-1,1:end-2); % difference of U in y-direction
            Vxi   = V(3:end,2:end-1)-V(1:end-2,2:end-1); % difference of V in xi-direction
            Vyi   = V(2:end-1,3:end)-V(2:end-1,1:end-2); % difference of V in y-direction
            
            dsxi  = abs(    (X(3:end,2:end-1)-X(1:end-2,2:end-1)) ...
                +1i*(Y(3:end,2:end-1)-Y(1:end-2,2:end-1)) );
            dsyi  = abs(    (X(2:end-1,3:end)-X(2:end-1,1:end-2)) ...
                +1i*(Y(2:end-1,3:end)-Y(2:end-1,1:end-2)) );
            vec_xi=  (    X(3:end,2:end-1)-X(1:end-2,2:end-1) ...
                +1i*(Y(3:end,2:end-1)-Y(1:end-2,2:end-1)) )./dsxi;
            vec_yi=  (      X(2:end-1,3:end)-X(2:end-1,1:end-2) ...
                +1i*(Y(2:end-1,3:end)-Y(2:end-1,1:end-2)) )./dsyi;
            Ux(2:end-1,2:end-1) = real(Uxi./dsxi.*vec_xi) ;%;+ real(Uyi./dsyi.*vec_yi);
%             Ux(2:end-1,2:end-1) = real(Uxi./dsxi.*vec_xi) + real(Uyi./dsyi.*vec_yi);
            Uy(2:end-1,2:end-1) = imag(Uxi./dsxi.*vec_xi) + imag(Uyi./dsyi.*vec_yi);
            Vx(2:end-1,2:end-1) = real(Vxi./dsxi.*vec_xi) ;%;+ real(Vyi./dsyi.*vec_yi);
%             Vx(2:end-1,2:end-1) = real(Vxi./dsxi.*vec_xi) + real(Vyi./dsyi.*vec_yi);
            Vy(2:end-1,2:end-1) = imag(Vxi./dsxi.*vec_xi) + imag(Vyi./dsyi.*vec_yi);
        else
            disp('No mean (U,V) velocity in reading data TWO-dimensional case');
            return
        end
        %---remove the ghost points (central difference)---1D----------------------
        Ux = Ux(2:end-1, 2:end-1);
        Vx = Vx(2:end-1, 2:end-1);
        Wx = Wx(2:end-1, 2:end-1);
        
        Uy = Uy(2:end-1, 2:end-1);
        Vy = Vy(2:end-1, 2:end-1);
        Wy = Wy(2:end-1, 2:end-1);
        
        Uz = Uz(2:end-1, 2:end-1);
        Vz = Vz(2:end-1, 2:end-1);
        Wz = Wz(2:end-1, 2:end-1);
    end
end


                 
end

