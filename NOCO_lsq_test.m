%% MKM: SYMBOLIC Full-Newton + Log-Scaling + Adaptive Kick
% Adapted for 10-State System (5 Surface, 5 Gas)
% Estimating 'A', 'b', 'Ea', and Sticking Coefficients
clear; clc; tic;
rng(42); 
%% ------------------------------------------------------------
% 0) Constants & Data
%% ------------------------------------------------------------
R = 1.987e-3; 
sden = 1.25e19 / 6.0222e23;
Data_raw = [
    428.0, 3013.8738, 3369.03658, 0.77481391, 6.49651665, 12.2314641;
    443.0, 3014.09234, 3369.2485, 6.72167625, 0, 6.72829859;
    463.0, 2985.75534, 3358.1031, 12.754629, 1.3046012, 18.4829541;
    483.0, 2986.03348, 3364.08969, 18.7610924, 1.57611719, 13.026145;
    503.0, 2951.94829, 3330.03099, 30.4958809, 0, 36.2308283;
    523.0, 2906.40645, 3301.66088, 168.565071, 2.17212789, 30.7872639;
    543.0, 2803.60123, 3250.41059, 375.062912, 36.826839, 94.0968451;
    563.0, 2454.45154, 3038.75394, 1034.1183, 60.0116553, 243.318058;
    583.0, 1406.4793, 2386.00302, 1847.83185, 151.942995, 724.79537;
    603.0, 123.625864, 1561.41559, 2082.99118, 300.0000000, 1274.99934;
    623.0, 15.026092, 1263.8142, 2713.36521, 501.966835, 1086.24937;
    643.0, 21.0722895, 708.477921, 3189.12082, 949.05433, 485.053376;
    663.0, 32.8004556, 239.033403, 3195.09417, 1476.35824, 38.5221584;
    683.0, 4.45021324, 222.113322, 3212.59039, 1533.91963, 15.9002411;
    703.0, 4.72172923, 233.861355, 42.2372917, 1516.99955, 21.9265715
];
Tvec = Data_raw(:, 1);
nc = numel(Tvec); ns = 10; 
nX = nc * ns; nB = 7; nE = 7; nA = 7; nS = 2;
nP = nX + nA + nB + nE + nS; nH = nc * 10;
num_pts = nc * 5; 
yTarget_Gas = Data_raw(:, 2:6); 

% --- PARAMETERS ---
A_base = 1e13 * sden;
Ahat_init = log10(A_base) * ones(7,1);
Ea_gt   = [26; 13; 27; 21; 13; 32; 23.7]; 
Ea_scale = 10;
Eahat_init = Ea_gt / Ea_scale;
b_init  = zeros(7,1); 
S_init  = 0.1 * ones(2,1); % Sticking coefficients starting at 0.1

% --- BOUNDS ---
Ahatlo = (log10(A_base) - 2) * ones(7,1); 
Ahatup = (log10(A_base) + 2) * ones(7,1);
blo     = -5.0 * ones(7,1);   
bup     =  2.0 * ones(7,1);
Eahatlo =  0.0 * ones(7,1); 
Eahatup = 10.0 * ones(7,1); 
Slo     = 0.01 * ones(2,1);
Sup     = 1.0  * ones(2,1);

%% ------------------------------------------------------------
% 1) Symbolic Setup
%% ------------------------------------------------------------
fprintf('Deriving Single-Point KKT System...\n');
x_s   = sym('x_s',   [1 10], 'real');
lam_s = sym('lam_s', [10 1], 'real');
T_s   = sym('T_s',   'real');
yT_s  = sym('yT_s',  [1 5],  'real');

Ahat_sym = sym('Ahat', [7 1], 'real');
b_sym    = sym('b_sym', [7 1], 'real');  
Eahat    = sym('Eahat', [7 1], 'real'); 
S_sym    = sym('S_sym', [2 1], 'real');
p_s      = [Ahat_sym; b_sym; Eahat; S_sym];

f_s = sym(0);
for i = 1:5 
    f_s = f_s + (x_s(i+5).*1e6./30 - yT_s(i))^2;
end
f_s = f_s / sym(nc * 5); 

tno = x_s(1); tn = x_s(2); tn2o = x_s(3); tco = x_s(4); to = x_s(5);
pno = x_s(6); pco = x_s(7); pco2 = x_s(8); pn2 = x_s(9); pn2o = x_s(10);
ts = 1 - sum(x_s(1:5)); 

k1 = S_sym(1) * sqrt(8.314462618e3 * T_s / (2 * pi * 30));
k7 = S_sym(2) * sqrt(8.314462618e3 * T_s / (2 * pi * 28));

k_est = (10.^Ahat_sym) .* (T_s.^b_sym) .* exp(-(Eahat * sym(Ea_scale)) / (sym(R) * T_s));
k2 = k_est(1); k3 = k_est(2); k4 = k_est(3); k5 = k_est(4); 
k6 = k_est(5); k8 = k_est(6); k9 = k_est(7);

r1 = k1 * pno * ts;        r2 = k2 * tno;        r3 = k3 * tno * ts;
r4 = k4 * (tn)^2;          r5 = k5 * tno * tn;   r6 = k6 * tn2o;
r7 = k7 * pco * ts;        r8 = k8 * tco;        r9 = k9 * tco * to;

dy1 = (r1 - r2 - r3 - r5);
dy2 = (r3 - 2*r4 - r5);
dy3 = (r5 - r6);
dy4 = (r7 - r8 - r9);
dy5 = (r3 - r9);

gas_scale = 1 / 0.47e6;
dy6  = gas_scale * ((3000*30*1e-6 - pno)/0.105) + (r2 - r1);
dy7  = gas_scale * ((3400*30*1e-6 - pco)/0.105) + (r8 - r7);
dy8  = gas_scale * ((0 - pco2)/0.105) + (r9);
dy9  = gas_scale * ((0 - pn2)/0.105) + (r4);
dy10 = gas_scale * ((0 - pn2o)/0.105) + (r6);
h_s = [dy1; dy2; dy3; dy4; dy5; dy6; dy7; dy8; dy9; dy10];

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
x0mat(:, 6:10) = yTarget_Gas.*30./1e6; 
p = [reshape(x0mat.',[],1); Ahat_init; b_init; Eahat_init; S_init];
lamk = zeros(nH,1);
p = project_all_bounds(p, nc, ns, nX, nA, nB, nE, Ahatlo, Ahatup, blo, bup, Eahatlo, Eahatup, Slo, Sup);
p_best = p; sse_best = inf; h_best = inf; r_best = inf;
best_h_vec = zeros(nH, 1); iter_best = 0;

%% ------------------------------------------------------------
% 3) PHASE 1: Full Newton Solver
%% ------------------------------------------------------------
fprintf('\n=== Full Newton Solve ===\n');
maxIter = 2500; 
last_maxr = inf; stag_count = 0;

for k = 1:maxIter
    [rk, Jk, hk, sse_total] = build_full_system(p, lamk, Tvec, yTarget_Gas, funcs, nc, ns, nX, nP, nH);
    maxr = max(abs(rk)); maxh = max(abs(hk));
    curr_sse = sse_total * num_pts;
    
    if maxh < 5e-4 && maxr < 5e-3 && curr_sse < sse_best
        sse_best = curr_sse; h_best = maxh; r_best = maxr;
        p_best = p; best_h_vec = hk; iter_best = k;
        if maxh <= 2e-6 && maxr < 2e-6, break; end
    end
    
    if abs(maxr - last_maxr) < 1e-12, stag_count = stag_count + 1; else, stag_count = 0; end
    last_maxr = maxr;
    
    if stag_count > 25
        p = p + (1e-5 * randn(size(p))); 
        lamk = 0.5 * lamk; stag_count = 0;
    end
    
    if maxh > 1e-3
        for kr = 1:20
            [~, Jh_full, hk_full, ~] = build_full_system(p, lamk, Tvec, yTarget_Gas, funcs, nc, ns, nX, nP, nH);
            Jh_x = Jh_full(nP+1:end, 1:nX); 
            dx = -1.0 * lsqminnorm(Jh_x, hk_full);
            p(1:nX) = p(1:nX) + dx;
            p = project_all_bounds(p, nc, ns, nX, nA, nB, nE, Ahatlo, Ahatup, blo, bup, Eahatlo, Eahatup, Slo, Sup);
            [~, ~, hk_chk, ~] = build_full_system(p, lamk, Tvec, yTarget_Gas, funcs, nc, ns, nX, nP, nH);
            if max(abs(hk_chk)) < 1e-6, break; end 
        end
    end
    
    dz = lsqminnorm(Jk, rk);
    alpha_try = 1.0; rho = 0.5; c1 = 1e-4; Phi0 = 0.5*(rk.'*rk);
    accepted = false;
    
    for bt = 1:30
        zk_try = [p; lamk] - alpha_try*dz;
        p_try = project_all_bounds(zk_try(1:nP), nc, ns, nX, nA, nB, nE, Ahatlo, Ahatup, blo, bup, Eahatlo, Eahatup, Slo, Sup);
        lam_try = zk_try(nP+1:end);
        [r_try, ~, ~, ~] = build_full_system(p_try, lam_try, Tvec, yTarget_Gas, funcs, nc, ns, nX, nP, nH);
        if 0.5*(r_try.'*r_try) <= Phi0*(1 - c1*alpha_try)
            p = p_try; lamk = lam_try; accepted = true; break;
        end
        alpha_try = alpha_try * rho;
    end
    if ~accepted, lamk = 0.8*lamk; end
    if mod(k,100) == 0 || k == 1
        fprintf('%5d | R:%10.2e H:%10.2e SSE:%10.2e\n', k, maxr, maxh, curr_sse);
    end
end

%% ------------------------------------------------------------
% 4) Final Report (Updated for A, b, Ea, and Sticking Coeffs)
%% ------------------------------------------------------------
p = p_best; 
fprintf('\n================ FINAL BEST RESULTS =================\n');

% Reshape states
Xfinal = reshape(p(1:nX), [ns nc]).';

% Extract Parameters from the end of the p vector
% Indexing: nX + (1:7) is Ahat, (8:14) is b, (15:21) is Eahat, (22:23) is S
A_opt     = 10.^p(nX + (1:7));
b_opt     = p(nX + 7 + (1:7)); 
Eahat_opt = p(nX + 14 + (1:7));
Ea_opt    = Eahat_opt * Ea_scale;
S_opt     = p(nX + 21 + (1:2));

fprintf('Iteration where best value was observed: %d\n', iter_best);
fprintf('Lowest Total SSE Found (Gas Species 6-10 only): %.4e\n', sse_best);
fprintf('Overall Max Absolute Constraint Violation (h_max): %.2e\n', h_best);
fprintf('Max KKT Residual (r_max) at Best: %.2e\n\n', r_best);

fprintf('--- Sticking Coefficients ---\n');
fprintf('S1 (NO Adsorption): %.6f\n', S_opt(1));
fprintf('S7 (CO Adsorption): %.6f\n\n', S_opt(2));

fprintf('--- Final Estimated Kinetic Parameters (A, b, Ea) ---\n');
StepNames = {'k2'; 'k3'; 'k4'; 'k5'; 'k6'; 'k8'; 'k9'};
ParamTable = table(StepNames, A_opt, b_opt, Ea_opt, ...
    'VariableNames', {'Rate','A_Estimated','b','Ea_kcal_mol'});
disp(ParamTable);

fprintf('\n--- Final Surface Coverages (y1 - y5) ---\n');
RowNamesT = arrayfun(@(t) sprintf('%.1f K', t), Tvec, 'UniformOutput', false);
SurfaceTable = array2table(Xfinal(:, 1:5), ...
    'VariableNames', {'tno', 'tn', 'tn2o', 'tco', 'to'}, 'RowNames', RowNamesT);
disp(SurfaceTable);

fprintf('\n--- Final Gas Mole Fractions (ppm/normalized) ---\n');
% Scale back to match the input data format if necessary
GasTable = array2table(Xfinal(:, 6:10).*1e6./30, ...
    'VariableNames', {'pno', 'pco', 'pco2', 'pn2', 'pn2o'}, 'RowNames', RowNamesT);
disp(GasTable);

fprintf('\n--- Max Absolute Constraint Violation Per Equation ---\n');
h_mat = reshape(best_h_vec, [10, nc]).';
max_h_per_eq = max(abs(h_mat), [], 1).';
EqNames = {'dy1 (tno)'; 'dy2 (tn)'; 'dy3 (tn2o)'; 'dy4 (tco)'; 'dy5 (to)'; ...
           'dy6 (pno)'; 'dy7 (pco)'; 'dy8 (pco2)'; 'dy9 (pn2)'; 'dy10 (pn2o)'};
MaxViolationTable = table(EqNames, max_h_per_eq, 'VariableNames', {'Equation', 'Max_Abs_Violation'});
disp(MaxViolationTable);

toc;

%% ------------------------------------------------------------
% 5) LOCAL FUNCTIONS
%% ------------------------------------------------------------
function [R, J, h_all, sse_total] = build_full_system(p, lam, Tvec, yTarget, funcs, nc, ns, nX, nP, nH)
    R = zeros(nP + nH, 1);
    J = spalloc(nP + nH, nP + nH, (nP+nH)*10); 
    h_all = zeros(nH, 1); sse_total = 0;
    p_params = p(nX+1 : nP); 
    for c = 1:nc
        ix = (c-1)*ns + 1 : c*ns;
        il = nP + (c-1)*10 + 1 : nP + c*10;
        ip = nX + 1 : nP;
        x_c = p(ix).'; lam_c = lam((c-1)*10 + 1 : c*10); T_c = Tvec(c); yT_c = yTarget(c, :);
        rx_val = funcs.rx(x_c, p_params, lam_c, T_c, yT_c);
        rp_val = funcs.rp(x_c, p_params, lam_c, T_c, yT_c);
        h_val = funcs.h(x_c, p_params, T_c);
        Jxx_val = funcs.Jxx(x_c, p_params, lam_c, T_c, yT_c);
        Jxp_val = funcs.Jxp(x_c, p_params, lam_c, T_c, yT_c);
        Jpp_val = funcs.Jpp(x_c, p_params, lam_c, T_c, yT_c);
        hx_val = funcs.hx(x_c, p_params, T_c);
        hp_val = funcs.hp(x_c, p_params, T_c);
        R(ix) = rx_val; R(ip) = R(ip) + rp_val; R(il) = h_val;
        h_all((c-1)*10 + 1 : c*10) = h_val;
        J(ix, ix) = Jxx_val; J(ix, ip) = Jxp_val; J(ip, ix) = Jxp_val.';
        J(ip, ip) = J(ip, ip) + Jpp_val; 
        J(il, ix) = hx_val; J(ix, il) = hx_val.'; J(il, ip) = hp_val; J(ip, il) = hp_val.';
        sse_total = sse_total + funcs.f(x_c, yT_c);
    end
end

function p = project_all_bounds(p, nc, ns, nX, nA, nB, nE, Ahatlo, Ahatup, blo, bup, Eahatlo, Eahatup, Slo, Sup)
    x_states = p(1:nX); X = reshape(x_states, [ns nc]).'; 
    for c = 1:nc
        X(c, 6:10) = min(max(X(c, 6:10), 0.0), 1.0);
        surf = min(max(X(c, 1:5), 0.0), 1.0);
        if sum(surf) > 1.0, surf = proj_capped_simplex_bisect(surf.', 1.0, 0.0, 1.0).'; end
        X(c, 1:5) = surf;
    end
    p(1:nX) = reshape(X.', [], 1);
    iA = nX + (1:nA); iB = nX+nA + (1:nB); iE = nX+nA+nB + (1:nE); iS = nX+nA+nB+nE + (1:2);
    p(iA) = min(max(p(iA), Ahatlo), Ahatup);
    p(iB) = min(max(p(iB), blo), bup); 
    p(iE) = min(max(p(iE), Eahatlo), Eahatup);
    p(iS) = min(max(p(iS), Slo), Sup);
end

function x = proj_capped_simplex_bisect(y, targetSum, lb, ub)
    tau_lo = min(y)-ub-1; tau_hi = max(y)-lb+1;
    for it = 1:50
        tau = 0.5*(tau_lo + tau_hi); x = min(max(y - tau, lb), ub);
        if sum(x) > targetSum, tau_lo = tau; else, tau_hi = tau; end
    end
    x = min(max(y - tau_hi, lb), ub);
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