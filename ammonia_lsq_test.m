%% MKM: SYMBOLIC Full-Newton + Log-Scaling + Adaptive Kick
% Adapted for 8-State System (5 Surface, 3 Gas)
% Estimating 'b', 'Ea', 'A', and Sticking Coefficients
clear; clc; tic;
rng(42); 

%% ------------------------------------------------------------
% 0) Constants & Data
%% ------------------------------------------------------------
R = 8.314; 
sden = 1.25e19 / 6.0222e23;

% Updated Data_raw based on image_f87b24.png
% Format: [Temperature(K), NH3, N2, H2]
Data_raw = [
    573, 1.42007755, 0.14498061, 0.43494184;
    623, 1.25972726, 0.18506818, 0.55520455;
    673, 1.13575206, 0.21606198, 0.64818595;
    723, 0.82451271, 0.29387182, 0.88161547;
    773, 0.57843168, 0.35539208, 1.06617624;
    823, 0.22643871, 0.44339032, 1.33017097;
    873, 0.10608919, 0.47347770, 1.42043311;
    923, 0.05173247, 0.48706688, 1.46120065;
    973, 0.02942082, 0.49264480, 1.47793439
];
Tvec = Data_raw(:, 1);
nc = numel(Tvec); ns = 8; nr_est = 9;
nX = nc * ns; nB = nr_est; nE = nr_est; nA = nr_est; nS = 2; % 9b, 9Ea, 9A, 2 sticks
nP = nX + nB + nE + nA + nS; nH = nc * ns;
num_pts = nc * 3; 

yTarget_Gas = Data_raw(:, 2:4); 

% --- INITIAL PARAMETERS ---
b_init      = zeros(nr_est,1);
Ea_init     = [82; 210; 78; 78; 48; 75; 75; 22; 156].*1e3; 
Ea_scale    = 1e4;
Eahat_init  = Ea_init / Ea_scale;
A_init      = 1e15 * ones(nr_est, 1); % Initial A at 1e13
S_init      = [0.01; 0.011]; % Initial sticking coeffs

% --- INDIVIDUAL BOUNDS ---
blo     = -5.0 * ones(nr_est,1);   bup     =  2.0 * ones(nr_est,1);
Eahatlo =  0.0 * ones(nr_est,1);   Eahatup = 16.0 * ones(nr_est,1); 
Alo     =  1e11 * ones(nr_est,1);  Aup     = 1e14 * ones(nr_est,1);
Slo     =  1e-2 * ones(2,1);        Sup     = 0.02 * ones(2,1);

%% ------------------------------------------------------------
% 1) Symbolic Setup
%% ------------------------------------------------------------
fprintf('Deriving Single-Point KKT System with 29 Estimated Parameters...\n');

x_s   = sym('x_s',   [1 ns], 'real');
lam_s = sym('lam_s', [ns 1], 'real');
T_s   = sym('T_s',   'real');
yT_s  = sym('yT_s',  [1 3],  'real');

b_sym     = sym('b_sym', [nB 1], 'real');  
Eahat_sym = sym('Eahat', [nE 1], 'real'); 
A_sym     = sym('A_sym', [nA 1], 'real');
S_sym     = sym('S_sym', [nS 1], 'real');
p_s       = [b_sym; Eahat_sym; A_sym; S_sym];

f_s = sym(0);
for i = 1:3 
    f_s = f_s + (x_s(i+5) - yT_s(i))^2;
end
f_s = f_s / sym(nc * 3); 

th2 = x_s(1); tn = x_s(2); tnh3 = x_s(3); tnh2 = x_s(4); tnh = x_s(5);
ph2 = x_s(6); pn2 = x_s(7); pnh3 = x_s(8);
ts = 1 - (th2 + tn + tnh3 + tnh2 + tnh); 

% Parametrized Adsorption (S_sym(1) for H2, S_sym(2) for NH3)
k1a = S_sym(1) * sqrt(R * T_s / (2 * pi * 2));  
k2a = 1e-6 * sqrt(R * T_s / (2 * pi * 28)); % Still fixed per prompt
k3a = S_sym(2) * sqrt(R * T_s / (2 * pi * 17)); 

% Parametrized Arrhenius (A_sym)
k_est = (A_sym .* sden) .* (T_s.^b_sym) .* exp(-(Eahat_sym * sym(Ea_scale)) / (sym(R) * T_s));

r1 = k1a * ph2 * ts^2;   r2 = k_est(1) * th2^2;      
r3 = k2a * pn2 * ts^2;   r4 = k_est(2) * tn^2;       
r5 = k3a * pnh3 * ts;    r6 = k_est(3) * tnh3;       
r7 = k_est(4) * tnh3 * ts; r8 = k_est(5) * tnh2 * th2; 
r9 = k_est(6) * tnh2 * ts; r10 = k_est(7) * tnh * th2; 
r11 = k_est(8) * tnh * ts; r12 = k_est(9) * tn * th2;  

d_th2  = 2*(r1 - r2) + (r7 - r8) + (r9 - r10) + (r11 - r12);
d_tn   = 2*(r3 - r4) + (r11 - r12);
d_tnh3 = (r5 - r6) - (r7 - r8);
d_tnh2 = (r7 - r8) - (r9 - r10);
d_tnh  = (r9 - r10) - (r11 - r12);

gas_scale = 1 / 150e3; 
tau = 1;            
d_ph2  = gas_scale * ((0 - ph2)/tau) + (r2 - r1);
d_pn2  = gas_scale * ((0 - pn2)/tau) + (r4 - r3);
d_pnh3 = gas_scale * ((2 - pnh3)/tau) + (r6 - r5);

h_s = [d_th2; d_tn; d_tnh3; d_tnh2; d_tnh; d_ph2; d_pn2; d_pnh3];
L_s = f_s + lam_s.' * h_s;

rx_s = gradient(L_s, x_s.');
rp_s = gradient(L_s, p_s);

funcs.rx  = matlabFunction(rx_s,  'Vars', {x_s, p_s, lam_s, T_s, yT_s});
funcs.rp  = matlabFunction(rp_s,  'Vars', {x_s, p_s, lam_s, T_s, yT_s});
funcs.h   = matlabFunction(h_s,   'Vars', {x_s, p_s, T_s});
funcs.f   = matlabFunction(f_s,   'Vars', {x_s, yT_s});
funcs.Jxx = matlabFunction(jacobian(rx_s, x_s.'), 'Vars', {x_s, p_s, lam_s, T_s, yT_s});
funcs.Jxp = matlabFunction(jacobian(rx_s, p_s),   'Vars', {x_s, p_s, lam_s, T_s, yT_s});
funcs.Jpp = matlabFunction(jacobian(rp_s, p_s),   'Vars', {x_s, p_s, lam_s, T_s, yT_s});
funcs.hx  = matlabFunction(jacobian(h_s, x_s.'), 'Vars', {x_s, p_s, T_s});
funcs.hp  = matlabFunction(jacobian(h_s, p_s),    'Vars', {x_s, p_s, T_s});

%% ------------------------------------------------------------
% 2) Initialization
%% ------------------------------------------------------------
x0mat = zeros(nc, ns);
x0mat(:, 1:5)  = 0.1; 
x0mat(:, 6:8) = yTarget_Gas; 
p = [reshape(x0mat.',[],1); b_init; Eahat_init; A_init; S_init];
lamk = zeros(nH,1);

p = project_all_bounds_updated(p, nc, ns, nX, nB, nE, nA, nS, blo, bup, Eahatlo, Eahatup, Alo, Aup, Slo, Sup);
p_best = p; sse_best = inf; h_best = inf; r_best = inf;
best_h_vec = zeros(nH, 1); iter_best = 0;

%% ------------------------------------------------------------
% 3) PHASE 1: Full Newton Solver (Unchanged Logic)
%% ------------------------------------------------------------
fprintf('\n=== Full Newton Log-Scaled Solve ===\n');
maxIter = 2500; last_maxr = inf; stag_count = 0;
for k = 1:maxIter
    [rk, Jk, hk, sse_total] = build_full_system(p, lamk, Tvec, yTarget_Gas, funcs, nc, ns, nX, nP, nH);
    maxr = max(abs(rk)); maxh = max(abs(hk));
    curr_sse = sse_total * num_pts;
    if maxh < 2e-5 && maxr < 2e-5 && curr_sse < 1e-5
        sse_best = curr_sse; h_best = maxh; r_best = maxr;
        p_best = p; best_h_vec = hk; iter_best = k;
        if maxh <= 2e-5 && maxr < 2e-5, break; end
    end
    if abs(maxr - last_maxr) < 1e-5, stag_count = stag_count + 1; else, stag_count = 0; end
    last_maxr = maxr;
    if stag_count > 25
        p = p + (1e-3 * randn(size(p))); lamk = 0.5 * lamk; stag_count = 0;
    end
    if maxh > 1e-3
        for kr = 1:20
            [~, Jh_full, hk_full, ~] = build_full_system(p, lamk, Tvec, yTarget_Gas, funcs, nc, ns, nX, nP, nH);
            Jh_x = Jh_full(nP+1:end, 1:nX); 
            dx = -1.0 * mylsqminnorm(Jh_x, hk_full);
            p(1:nX) = p(1:nX) + dx;
            p = project_all_bounds_updated(p, nc, ns, nX, nB, nE, nA, nS, blo, bup, Eahatlo, Eahatup, Alo, Aup, Slo, Sup);
            [~, ~, hk_chk, ~] = build_full_system(p, lamk, Tvec, yTarget_Gas, funcs, nc, ns, nX, nP, nH);
            if max(abs(hk_chk)) < 1e-6, break; end 
        end
    end
    dz = mylsqminnorm(Jk, rk);
    alpha_try = 1.0; rho = 0.5; Phi0 = 0.5*(rk.'*rk); accepted = false;
    for bt = 1:30
        zk_try = [p; lamk] - alpha_try*dz;
        p_try = project_all_bounds_updated(zk_try(1:nP), nc, ns, nX, nB, nE, nA, nS, blo, bup, Eahatlo, Eahatup, Alo, Aup, Slo, Sup);
        lam_try = zk_try(nP+1:end);
        [r_try, ~, ~, ~] = build_full_system(p_try, lam_try, Tvec, yTarget_Gas, funcs, nc, ns, nX, nP, nH);
        if 0.5*(r_try.'*r_try) <= Phi0, p = p_try; lamk = lam_try; accepted = true; break; end
        alpha_try = alpha_try * rho;
    end
    if ~accepted, lamk = 0.8*lamk; end
    if mod(k,100) == 0 || k == 1, fprintf('%5d | R:%10.2e H:%10.2e SSE:%10.2e\n', k, maxr, maxh, curr_sse); end
end

%% ------------------------------------------------------------
% 4) Final Report (Updated for 29 Estimated Parameters)
%% ------------------------------------------------------------
p = p_best; 
fprintf('\n================ FINAL BEST RESULTS =================\n');
Xfinal = reshape(p(1:nX), [ns nc]).';

% Correct Unpacking for the 29 parameters
% Order: states(nX), b(9), Ea(9), A(9), S(2)
b_opt  = p(nX+1 : nX+nB);
Ea_opt = p(nX+nB+1 : nX+nB+nE) * Ea_scale;
A_opt  = p(nX+nB+nE+1 : nX+nB+nE+nA);
S_opt  = p(nX+nB+nE+nA+1 : nP);

fprintf('Iteration where best value was observed: %d\n', iter_best);
fprintf('Lowest Total SSE Found (Gas Species 6-8 only): %.4e\n', sse_best);
fprintf('Overall Max Absolute Constraint Violation (h_max): %.2e\n', h_best);
fprintf('Max KKT Residual (r_max) at Best: %.2e\n\n', r_best);

fprintf('--- Optimized Adsorption (Sticking Coefficients) ---\n');
fprintf('H2 Sticking Coefficient:  %.4f\n', S_opt(1));
fprintf('NH3 Sticking Coefficient: %.4f\n\n', S_opt(2));

fprintf('--- Max Absolute Constraint Violation Per Equation ---\n');
h_mat = reshape(best_h_vec, [ns, nc]).';
max_h_per_eq = max(abs(h_mat), [], 1).';
EqNames = {'dy1 (th2)'; 'dy2 (tn)'; 'dy3 (tnh3)'; 'dy4 (tnh2)'; 'dy5 (tnh)'; ...
           'dy6 (ph2)'; 'dy7 (pn2)'; 'dy8 (pnh3)'};
MaxViolationTable = table(EqNames, max_h_per_eq, 'VariableNames', {'Equation', 'Max_Abs_Violation'});
disp(MaxViolationTable);

fprintf('\n--- Final Estimated Kinetics (A, b, Ea) ---\n');
StepNames = {'k_est1'; 'k_est2'; 'k_est3'; 'k_est4'; 'k_est5'; 'k_est6'; 'k_est7'; 'k_est8'; 'k_est9'};
ParamTable = table(StepNames, A_opt, b_opt, Ea_opt, ...
    'VariableNames', {'Rate','A_Optimized','b','Ea_kcal_mol'});
disp(ParamTable);

fprintf('\n--- Final Surface Coverages (y1 - y5) ---\n');
RowNamesT = arrayfun(@(t) sprintf('%.1f K', t), Tvec, 'UniformOutput', false);
SurfaceTable = array2table(Xfinal(:, 1:5), 'VariableNames', {'th2', 'tn', 'tnh3', 'tnh2', 'tnh'}, 'RowNames', RowNamesT);
disp(SurfaceTable);

fprintf('\n--- Final Gas Concentrations (y6 - y8) ---\n');
GasTable = array2table(Xfinal(:, 6:8), 'VariableNames', {'NH3', 'N2', 'H2'}, 'RowNames', RowNamesT);
disp(GasTable);
toc;

%% ------------------------------------------------------------
% 6) Visual Confirmation
%% ------------------------------------------------------------
figure('Color', 'w', 'Position', [100, 100, 1000, 400]);
% Updated titles to match the ammonia system species
species_names = {'NH_3 (y6)', 'N_2 (y7)', 'H_2 (y8)'};
colors = lines(3);

for i = 1:3
    subplot(1, 3, i);
    % Plot experimental targets
    plot(Tvec, yTarget_Gas(:, i), 'o', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', 'DisplayName', 'Data');
    hold on;
    % Plot model prediction (States 6, 7, and 8)
    plot(Tvec, Xfinal(:, i+5), '-', 'LineWidth', 2, 'Color', colors(i,:), 'DisplayName', 'Model');
    title(species_names{i});
    xlabel('Temperature (K)');
    ylabel('Concentration (mol/m^3)');
    grid on;
    legend('Location', 'best');
end
sgtitle('Ammonia System: KKT Model vs. Experimental Targets (Variable Volume)');

%% ------------------------------------------------------------
% 5) LOCAL FUNCTIONS
%% ------------------------------------------------------------
function [R, J, h_all, sse_total] = build_full_system(p, lam, Tvec, yTarget, funcs, nc, ns, nX, nP, nH)
    R = zeros(nP + nH, 1);
    J = spalloc(nP + nH, nP + nH, (nP+nH)*15); 
    h_all = zeros(nH, 1); sse_total = 0;
    p_params = p(nX+1 : nP); 
    for c = 1:nc
        ix = (c-1)*ns + 1 : c*ns; il = nP + (c-1)*ns + 1 : nP + c*ns; ip = nX + 1 : nP;
        x_c = p(ix).'; lam_c = lam((c-1)*ns + 1 : c*ns); T_c = Tvec(c); yT_c = yTarget(c, :);
        R(ix) = funcs.rx(x_c, p_params, lam_c, T_c, yT_c);
        R(ip) = R(ip) + funcs.rp(x_c, p_params, lam_c, T_c, yT_c);
        h_val = funcs.h(x_c, p_params, T_c); R(il) = h_val; h_all((c-1)*ns + 1 : c*ns) = h_val;
        J(ix, ix) = funcs.Jxx(x_c, p_params, lam_c, T_c, yT_c);
        Jxp = funcs.Jxp(x_c, p_params, lam_c, T_c, yT_c); J(ix, ip) = Jxp; J(ip, ix) = Jxp.';
        J(ip, ip) = J(ip, ip) + funcs.Jpp(x_c, p_params, lam_c, T_c, yT_c);
        hx = funcs.hx(x_c, p_params, T_c); J(il, ix) = hx; J(ix, il) = hx.';
        hp = funcs.hp(x_c, p_params, T_c); J(il, ip) = hp; J(ip, il) = hp.';
        sse_total = sse_total + funcs.f(x_c, yT_c);
    end
end

function p = project_all_bounds_updated(p, nc, ns, nX, nB, nE, nA, nS, blo, bup, Eahatlo, Eahatup, Alo, Aup, Slo, Sup)
    X = reshape(p(1:nX), [ns nc]).'; 
    for c = 1:nc
        X(c, 6:8) = min(max(X(c, 6:8), 0.0), 2.0);
        surf = min(max(X(c, 1:5), 0.0), 1.0);
        if sum(surf) > 1.0, surf = surf/sum(surf); end
        X(c, 1:5) = surf;
    end
    p(1:nX) = reshape(X.', [], 1);
    p(nX+1 : nX+nB) = min(max(p(nX+1 : nX+nB), blo), bup);
    p(nX+nB+1 : nX+nB+nE) = min(max(p(nX+nB+1 : nX+nB+nE), Eahatlo), Eahatup);
    p(nX+nB+nE+1 : nX+nB+nE+nA) = min(max(p(nX+nB+nE+1 : nX+nB+nE+nA), Alo), Aup);
    p(nX+nB+nE+nA+1 : end) = min(max(p(nX+nB+nE+nA+1 : end), Slo), Sup);
end

function x = mylsqminnorm(A, b)
    % MYLSQMINNORM: Custom minimum-norm least-squares solver using SVD.
    % Uses Absolute Hard Truncation to handle extremely poorly scaled KKT matrices
    % without introducing Tikhonov bias.
    
    if issparse(A)
        A = full(A);
    end
    
    [U, S, V] = svd(A, 'econ');
    s = diag(S);
    
    % --- THE FIX: Absolute Tolerance ---
    % We do not use max(s) because your A_init = 1e15 forces relative 
    % tolerances to be too large (e.g., 0.1), which deletes your concentration math.
    % We use an absolute floor to only drop true floating-point noise.
    tol = 1e-4; 
    
    s_inv = zeros(size(s));
    for i = 1:length(s)
        if s(i) > tol
            s_inv(i) = 1 / s(i);
        end
    end
    
    % Compute the exact minimum norm step (no bias!)
    x = V * diag(s_inv) * U' * b;
end