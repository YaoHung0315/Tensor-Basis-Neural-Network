function [Uxx, Vxx, Wxx, Uyy, Vyy, Wyy, Uzz, Vzz, Wzz] = get_velocity_gradient_2D( X, Y, Z, U, V, W)

% The mean flow velocity can be 1 or 2 dimensions.
Uxx = zeros(size(U));
Uyy = zeros(size(U));
Uzz = zeros(size(U));
Vxx = zeros(size(U));
Vyy = zeros(size(U));
Vzz = zeros(size(U));
Wxx = zeros(size(U));
Wyy = zeros(size(U));
Wzz = zeros(size(U));

% Mean flow 2D
    disp('sy==2');

    if 1;~isempty(X) && ~isempty(Y) && ~isempty(Z);
        if ~isempty(U) && ~isempty(V)

            Uxx = zeros(size(U));
            Uyy = zeros(size(U));

            Uxi   = U(3:end,2:end-1)-U(1:end-2,2:end-1); % difference of U in xi-direction
            Uyi   = U(2:end-1,3:end)-U(2:end-1,1:end-2); % difference of U in y-direction
            Vxi   = V(3:end,2:end-1)-V(1:end-2,2:end-1); % difference of V in xi-direction
            Vyi   = V(2:end-1,3:end)-V(2:end-1,1:end-2); % difference of V in y-direction
           
            dsxj  = (X(2:end-1,3:end)-X(2:end-1,1:end-2));
            dsyi  = (Y(3:end,2:end-1)-Y(1:end-2,2:end-1));

            Uxx(2:end-1,2:end-1) = (Uxi./dsxj);
            Uyy(2:end-1,2:end-1) = (Uyi./dsyi);
            Vxx(2:end-1,2:end-1) = (Vxi./dsxj);
            Vyy(2:end-1,2:end-1) = (Vyi./dsyi);

        else
            disp('No mean (U,V) velocity in reading data TWO-dimensional case');
            return
        end
        %---remove the ghost points (central difference)---1D----------------------
        Uxx = Uxx(2:end-1, 2:end-1);
        Vxx = Vxx(2:end-1, 2:end-1);
        Wxx = Wxx(2:end-1, 2:end-1);
        
        Uyy = Uyy(2:end-1, 2:end-1);
        Vyy = Vyy(2:end-1, 2:end-1);
        Wyy = Wyy(2:end-1, 2:end-1);
        
        Uzz = Uzz(2:end-1, 2:end-1);
        Vzz = Vzz(2:end-1, 2:end-1);
        Wzz = Wzz(2:end-1, 2:end-1);
    end



                 
end

