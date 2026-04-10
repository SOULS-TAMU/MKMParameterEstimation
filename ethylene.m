%% MKM: SYMBOLIC Full-Newton + Adaptive Kick (CSTR System)
% Update: Reverted Objective Scaling, Safe Rate Laws, Multi-Start CSV
clear; clc; tic;

%% ------------------------------------------------------------
% 0) Constants, Data, & Exact GAMS Bounds
%% ------------------------------------------------------------
R    = 8.314;       
tau  = 10;          
Cin  = [1.0; 0; 0.5; 0; 0.5; 0; 0]; 
Tvec = [835; 848; 861; 873];
nc   = numel(Tvec); 
ns   = 7; 
nr   = 3; 
nX   = nc*ns; 
nA   = nr; nE = nr; nKc = nr;
nTheta = nA + nE + nKc;
nP   = nX + nTheta; 
nH   = nc*ns; 
Ntot = nP + nH;
num_pts = nc * ns; 

xTarget = [ ...
    3.6748e-01, 4.1661e-01, 4.2497e-01, 4.3182e-01, 2.2427e-01, 2.7573e-01, 2.7573e-01;
    3.6626e-01, 4.1783e-01, 4.2596e-01, 4.3182e-01, 2.2404e-01, 2.7596e-01, 2.7596e-01;
    3.6534e-01, 4.1875e-01, 4.2670e-01, 4.3181e-01, 2.2385e-01, 2.7615e-01, 2.7615e-01;
    3.6470e-01, 4.1940e-01, 4.2721e-01, 4.3182e-01, 2.2373e-01, 2.7627e-01, 2.7627e-01 ];

% --- GROUND TRUTH ---
A_gt  = [1e8; 5e7; 1e7];           
Ea_gt = [120000; 100000; 80000];   
Kc_gt = [0.5; 1.2; 0.8];

% --- EXACT GAMS BOUNDS ---
Ea_scale = 100000; 

Alo  = [5e7; 2.5e7; 5e6];        Aup  = [1.5e8; 7.5e7; 1.5e7]; %50%A
% Alo  = [80000000; 40000000; 8000000];        Aup  = [120000000; 60000000; 12000000];
Ealo = [100000; 80000; 60000];   Eaup = [140000; 120000; 100000];
Kclo = [0.05; 0.12; 0.8];       Kcup = [5; 12; 8];
% Kclo = [0.4; 0.96; 0.64];       Kcup = [.6; 1.44; .96];

% Transformed Bounds for Solver
lnAlo   = log(Alo);              lnAup   = log(Aup);
Eahatlo = Ealo / Ea_scale;       Eahatup = Eaup / Ea_scale; 

%% ------------------------------------------------------------
% 1) Fast Symbolic Setup (Exact GAMS Match)
%% ------------------------------------------------------------
fprintf('Deriving Compact Single-T KKT System...\n');

xs     = sym('xs',     [1 ns], 'real');
xTarg  = sym('xTarg',  [1 ns], 'real');
Ts     = sym('Ts',     'real');
lam_s  = sym('lam_s',  [ns 1], 'real');
theta  = sym('theta',  [nTheta 1], 'real'); 

lnA_s   = theta(1:nA);
Eahat_s = theta(nA+1 : nA+nE);
Kc_s    = theta(nA+nE+1 : end);

% Forward rate constants
k_fwd = exp(lnA_s) .* exp(-(Eahat_s * sym(Ea_scale)) / (sym(R) * Ts));

% Net rates formulated identically to GAMS (distributing $x_1$ to avoid NaN from division by zero)
r1 = k_fwd(1) * (xs(1)       - (xs(2)*xs(3))/Kc_s(1));
r2 = k_fwd(2) * (xs(1)*xs(3) - (xs(4)^2)/Kc_s(2));
r3 = k_fwd(3) * (xs(5)*xs(3) - (xs(6)*xs(7))/Kc_s(3));

% CSTR Mass Balances
h_s = [ (sym(Cin(1)) - xs(1))/sym(tau) - r1 - r2;
        (sym(Cin(2)) - xs(2))/sym(tau) + r1;
        (sym(Cin(3)) - xs(3))/sym(tau) + r1 - r2 - r3;
        (sym(Cin(4)) - xs(4))/sym(tau) + 2*r2;
        (sym(Cin(5)) - xs(5))/sym(tau) - r3;
        (sym(Cin(6)) - xs(6))/sym(tau) + r3;
        (sym(Cin(7)) - xs(7))/sym(tau) + r3 ];

% Objective (Reverted to your original division by num_pts)
f_s = sum((xs - xTarg).^2) / sym(num_pts);

% Lagrangian
L_s = f_s + lam_s.' * h_s;

fprintf('Calculating Exact Block Derivatives...\n');
grad_x = jacobian(L_s, xs).';      
grad_t = jacobian(L_s, theta).';   
H_xx   = jacobian(grad_x, xs);     
H_xt   = jacobian(grad_x, theta);  
H_tt   = jacobian(grad_t, theta);  
Jh_x   = jacobian(h_s, xs);        
Jh_t   = jacobian(h_s, theta);     

fprintf('Compiling fast MATLAB functions...\n');
vars = {xs, Ts, xTarg, theta, lam_s};
fh_fun  = matlabFunction(h_s,    'Vars', vars);
gx_fun  = matlabFunction(grad_x, 'Vars', vars);
gt_fun  = matlabFunction(grad_t, 'Vars', vars);
Hxx_fun = matlabFunction(H_xx,   'Vars', vars);
Hxt_fun = matlabFunction(H_xt,   'Vars', vars);
Htt_fun = matlabFunction(H_tt,   'Vars', vars);
Jhx_fun = matlabFunction(Jh_x,   'Vars', vars);
Jht_fun = matlabFunction(Jh_t,   'Vars', vars);

%% ------------------------------------------------------------
% 2) Load CSV and Initialize Monte Carlo Storage
%% ------------------------------------------------------------
fprintf('\nLoading "initial_guesses.csv"...\n');
guess_data = readtable('initial_guesses.csv');
num_iters = 1000; 

guess_A_mat  = reshape(guess_data{:, 3}, nr, num_iters).';
guess_Ea_mat = reshape(guess_data{:, 4}, nr, num_iters).';
guess_Kc_mat = reshape(guess_data{:, 5}, nr, num_iters).';

all_SSE = zeros(num_iters, 1);
all_Dev_A  = zeros(num_iters, nr);
all_Dev_Ea = zeros(num_iters, nr);
all_Dev_Kc = zeros(num_iters, nr);

% Prepare Output Results File
outFile = fopen('results_iterations_matlab.csv', 'w');
fprintf(outFile, 'Iteration,SSE,r1_A,r1_Ea,r1_Kc,r2_A,r2_Ea,r2_Kc,r3_A,r3_Ea,r3_Kc\n');

fprintf('=== PHASE 1: Multi-Start Full Newton Solve ===\n');
for iter = 1:num_iters
    
    A_guess  = guess_A_mat(iter, :).';
    Ea_guess = guess_Ea_mat(iter, :).';
    Kc_guess = guess_Kc_mat(iter, :).';
    
    lnA_guess   = log(A_guess);
    Eahat_guess = Ea_guess / Ea_scale; 
    
    x0mat = xTarget; 
    p = [reshape(x0mat.',[],1); lnA_guess; Eahat_guess; Kc_guess];
    lamk = zeros(nH,1);
    
    p = project_all_bounds(p, nc, ns, nX, nA, nE, nKc, lnAlo, lnAup, Eahatlo, Eahatup, Kclo, Kcup);
    
    p_best = p;
    sse_best = inf;
    
    % --- NUMERICAL ASSEMBLY OF FULL SYSTEM ---
    maxIter = 2000; 
    last_maxr = inf; stag_count = 0;
    
    for k = 1:maxIter
        [rk, Jk, hk, Jh_k, curr_sse_scaled] = assemble_KKT(p, lamk, Tvec, xTarget, ...
            fh_fun, gx_fun, gt_fun, Hxx_fun, Hxt_fun, Htt_fun, Jhx_fun, Jht_fun, ...
            nc, ns, nTheta, nP, nH, Ntot, num_pts);
            
        maxr = max(abs(rk)); 
        maxh = max(abs(hk));
        
        % Restored your target scaling for objective termination
        curr_sse = curr_sse_scaled * num_pts; 
        
        if maxh < 1e-5 && maxr < 1e-5
            sse_best = curr_sse; p_best = p;
            break; 
        end
        
        if abs(maxr - last_maxr) < 1e-6, stag_count = stag_count + 1;
        else, stag_count = 0; end
        last_maxr = maxr;
        
        if stag_count > 25
            p = p + (1e-3 * randn(size(p))); 
            lamk = 0.5 * lamk; stag_count = 0;
        end
        
        if maxh > 1e-3
            for kr = 1:15
                [~, ~, hk_inner, Jh_inner, ~] = assemble_KKT(p, lamk, Tvec, xTarget, ...
                    fh_fun, gx_fun, gt_fun, Hxx_fun, Hxt_fun, Htt_fun, Jhx_fun, Jht_fun, ...
                    nc, ns, nTheta, nP, nH, Ntot, num_pts);
                    
                p = p - 0.5 * lsqminnorm(Jh_inner, hk_inner);
                p = project_all_bounds(p, nc, ns, nX, nA, nE, nKc, lnAlo, lnAup, Eahatlo, Eahatup, Kclo, Kcup);
                
                [~, ~, hk_check, ~, ~] = assemble_KKT(p, lamk, Tvec, xTarget, ...
                    fh_fun, gx_fun, gt_fun, Hxx_fun, Hxt_fun, Htt_fun, Jhx_fun, Jht_fun, ...
                    nc, ns, nTheta, nP, nH, Ntot, num_pts);
                if max(abs(hk_check)) < 1e-7, break; end
            end
        end
        
        dz = lsqminnorm(Jk, rk);
        alpha_try = 1.0; rho = 0.5; c1 = 1e-4; Phi0 = 0.5*(rk.'*rk);
        accepted = false;
        
        for bt = 1:30
            zk_try = [p; lamk] - alpha_try*dz;
            p_try = project_all_bounds(zk_try(1:nP), nc, ns, nX, nA, nE, nKc, lnAlo, lnAup, Eahatlo, Eahatup, Kclo, Kcup);
            lam_try = zk_try(nP+1:end);
            
            [r_try, ~, ~, ~, ~] = assemble_KKT(p_try, lam_try, Tvec, xTarget, ...
                fh_fun, gx_fun, gt_fun, Hxx_fun, Hxt_fun, Htt_fun, Jhx_fun, Jht_fun, ...
                nc, ns, nTheta, nP, nH, Ntot, num_pts);
                
            if 0.5*(r_try.'*r_try) <= Phi0*(1 - c1*alpha_try)
                p = p_try; lamk = lam_try; accepted = true; break;
            end
            alpha_try = alpha_try * rho;
        end
        
        if ~accepted, lamk = 0.8*lamk; end
        
        if maxh < 1e-5 && curr_sse < sse_best
            sse_best = curr_sse; p_best = p;
        end
    end
    
    A_opt  = exp(p_best(nX+(1:nA))); 
    Ea_opt = p_best(nX+nA+(1:nE)) * Ea_scale;
    Kc_opt = p_best(nX+nA+nE+(1:nKc));
    
    all_SSE(iter) = sse_best;
    all_Dev_A(iter, :)  = abs(1 - A_gt ./ A_opt) * 100;
    all_Dev_Ea(iter, :) = abs(1 - Ea_gt ./ Ea_opt) * 100;
    all_Dev_Kc(iter, :) = abs(1 - Kc_gt ./ Kc_opt) * 100;
    
    fprintf(outFile, '%d,%.8e,%.5e,%.5f,%.5f,%.5e,%.5f,%.5f,%.5e,%.5f,%.5f\n', ...
        iter, sse_best, ...
        A_opt(1), Ea_opt(1), Kc_opt(1), ...
        A_opt(2), Ea_opt(2), Kc_opt(2), ...
        A_opt(3), Ea_opt(3), Kc_opt(3));

    if mod(iter, 50) == 0 || iter == 1
        fprintf('Completed %4d / %d runs...\n', iter, num_iters);
    end
end
fclose(outFile);
fprintf('Saved iteration data to results_iterations_matlab.csv\n');

%% ------------------------------------------------------------
% 3) Final Averaged Report
%% ------------------------------------------------------------
fprintf('\n================ FINAL AVERAGED RESULTS =================\n');
Avg_SSE = mean(all_SSE);
Avg_Dev_A  = mean(all_Dev_A);
Avg_Dev_Ea = mean(all_Dev_Ea);
Avg_Dev_Kc = mean(all_Dev_Kc);

fprintf('Average Total SSE Over %d Runs: %.6e\n\n', num_iters, Avg_SSE);

StepNames = {'rxn1'; 'rxn2'; 'rxn3'};
AveragesTable = table(StepNames, Avg_Dev_A.', Avg_Dev_Ea.', Avg_Dev_Kc.', ...
    'VariableNames', {'Reaction_Step', 'Avg_Dev_A_percent', 'Avg_Dev_Ea_percent', 'Avg_Dev_Kc_percent'});
disp(AveragesTable);
toc;

%% ------------------------------------------------------------
% 4) LOCAL FUNCTIONS
%% ------------------------------------------------------------
function [R, J, h_val, Jh_full, sse] = assemble_KKT(p, lam, Tvec, xTarget, ...
    fh_fun, gx_fun, gt_fun, Hxx_fun, Hxt_fun, Htt_fun, Jhx_fun, Jht_fun, ...
    nc, ns, nTheta, nP, nH, Ntot, num_pts)
    
    R = zeros(Ntot, 1);
    J = zeros(Ntot, Ntot);
    Jh_full = zeros(nH, nP); 
    h_val = zeros(nH, 1);
    sse = 0;
    
    nX = nc * ns;
    theta = p(nX+1 : nP);
    X = reshape(p(1:nX), [ns, nc]).'; 
    
    for i = 1:nc
        idx_x   = (i-1)*ns + 1 : i*ns;
        idx_t   = nX + 1 : nX + nTheta;
        idx_lam = nP + (i-1)*ns + 1 : nP + i*ns;
        
        xs_i    = X(i, :);
        xTarg_i = xTarget(i, :);
        Ts_i    = Tvec(i);
        lam_i   = lam(idx_lam - nP);
        
        h_i   = fh_fun(xs_i, Ts_i, xTarg_i, theta, lam_i);
        gx_i  = gx_fun(xs_i, Ts_i, xTarg_i, theta, lam_i);
        gt_i  = gt_fun(xs_i, Ts_i, xTarg_i, theta, lam_i);
        Hxx_i = Hxx_fun(xs_i, Ts_i, xTarg_i, theta, lam_i);
        Hxt_i = Hxt_fun(xs_i, Ts_i, xTarg_i, theta, lam_i);
        Htt_i = Htt_fun(xs_i, Ts_i, xTarg_i, theta, lam_i);
        Jhx_i = Jhx_fun(xs_i, Ts_i, xTarg_i, theta, lam_i);
        Jht_i = Jht_fun(xs_i, Ts_i, xTarg_i, theta, lam_i);
        
        R(idx_x)   = gx_i;                      
        R(idx_t)   = R(idx_t) + gt_i;           
        R(idx_lam) = h_i;                       
        
        J(idx_x, idx_x) = Hxx_i;
        J(idx_x, idx_t) = Hxt_i;
        J(idx_t, idx_x) = Hxt_i.';              
        J(idx_t, idx_t) = J(idx_t, idx_t) + Htt_i;
        
        J(idx_x, idx_lam) = Jhx_i.';
        J(idx_lam, idx_x) = Jhx_i;
        J(idx_t, idx_lam) = J(idx_t, idx_lam) + Jht_i.';
        J(idx_lam, idx_t) = Jht_i;
        
        Jh_full(idx_lam - nP, idx_x) = Jhx_i;
        Jh_full(idx_lam - nP, idx_t) = Jht_i;
        
        h_val(idx_lam - nP) = h_i;
        
        % Reverted to your averaged SSE
        sse = sse + sum((xs_i - xTarg_i).^2) / num_pts;
    end
end

function p = project_all_bounds(p, nc, ns, nX, nA, nE, nKc, lnAlo, lnAup, Eahatlo, Eahatup, Kclo, Kcup)
    p(1:nX) = min(max(p(1:nX), 1e-12), 10.0); 
    
    iA  = nX + (1:nA); 
    iE  = nX+nA + (1:nE); 
    iKc = nX+nA+nE + (1:nKc);
    
    p(iA)  = min(max(p(iA), lnAlo), lnAup); 
    p(iE)  = min(max(p(iE), Eahatlo), Eahatup);
    p(iKc) = min(max(p(iKc), Kclo), Kcup);
end