% V_voxel: The volume of the voxel (cuboid).
% S_voxel_pre and S_voxel_post: The signal of the voxel before and after injection.
% S_blood_pre and S_blood_post: The signal of the blood vessel before and after injection.
% The system of nonlinear equations (equations) is set up with two unknowns: 
% 
% S_tissue (signal of tissue) and
% r_blood_vessel (radius of the blood vessel).
% 
% Initial Guess: You provide an initial guess for both unknowns, which is required by the fsolve function.
% fsolve Solver: This function iteratively solves the system of nonlinear equations using numerical methods.




function [S_tissue, r_blood_vessel] = PVT(V_voxel, S_voxel_pre, S_voxel_post, S_blood_pre, S_blood_post)

    % Define constants
    h_voxel = V_voxel^(1/3);  % Approximate height of voxel assuming it's cubic for simplicity

    % Define the system of nonlinear equations
    % The equations are nonlinear because of the r_blood_vessel^2 term (volume of blood vessel).
    equations = @(x) [
        (V_voxel - pi * (x(2)^2) * h_voxel) * x(1) + pi * (x(2)^2) * h_voxel * S_blood_pre - V_voxel * S_voxel_pre;
        (V_voxel - pi * (x(2)^2) * h_voxel) * x(1) + pi * (x(2)^2) * h_voxel * S_blood_post - V_voxel * S_voxel_post;
    ];
    
    % Initial guess for the unknowns [S_tissue, r_blood_vessel]
    initial_guess = [100, 0.01];  % Initial guess for signal of tissue and blood vessel radius

    % Solve the system of nonlinear equations using fsolve
    options = optimoptions('fsolve', 'Display', 'iter');  % Display iteration output
    [solution, ~] = fsolve(equations, initial_guess, options);

    % Extract the solution values
    S_tissue = solution(1);
    r_blood_vessel = solution(2);

    % Display the results
    disp(['Signal of Tissue: ', num2str(S_tissue)]);
    disp(['Radius of Blood Vessel: ', num2str(r_blood_vessel)]);
end
