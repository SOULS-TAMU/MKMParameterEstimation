%% MKM: SYMBOLIC Full-Newton + Log-Scaling + Adaptive Kick
% Adapted for 16-State Methanol System (11 Surface, 5 Gas)
% Estimating 'b', 'Ea', 'A', and 5 Individual Sticking Coefficients
clear; clc; tic;
rng(42); 
%% ------------------------------------------------------------
% 0) Constants & Data
%% ------------------------------------------------------------
R = 8.314; 
sden = 1.25e19 / 6.0222e23;
% Data_raw: [Temperature, H2, CO2, H2O, CO, CH3OH]
Data_raw = [
    623, 404.51683181, 21.00168532,  2.48471467, 20.75588789, 1.72882678;
    673, 359.79726987,  3.87656422, 69.60983578, 69.08378855, 0.526047231;
    723, 356.14698237,  0.01498587,  3.47141427,  3.05101244, 0.420401755
];
Tvec = Data_raw(:, 1);
nc = numel(Tvec); ns = 16; nr_est = 20; % 20 Arrhenius Rates
nX = nc * ns; nB = nr_est; nE = nr_est; nA = nr_est; nS = 5; % 5 Sticking Coeffs
nP = nX + nB + nE + nA + nS; nH = nc * ns;
num_pts = nc * 5; 
yTarget_Gas = Data_raw(:, 2:6); 
% --- INITIAL PARAMETERS ---
b_init      = zeros(nr_est,1);
Ea_val      = 80000; % Initial Ea guess
Ea_init     = Ea_val * ones(nr_est,1); 
Ea_scale    = 1e4;
Eahat_init  = Ea_init / Ea_scale;
lnA_init    = log(1e13) * ones(nr_est, 1); 
S_init      = 0.1 * ones(nS, 1); % Initial sticking coeffs (Vector of 5)
% --- INDIVIDUAL BOUNDS ---
blo      = -5.0 * ones(nr_est,1);    bup      =  2.0 * ones(nr_est,1);
Eahatlo  = (60000/Ea_scale) * ones(nr_est,1);    
Eahatup  = (100000/Ea_scale) * ones(nr_est,1); 
lnAlo    = log(1e11) * ones(nr_est,1); lnAup    = log(1e14) * ones(nr_est,1); 
Slo      = 0.01 * ones(nS, 1);         Sup      = 1.0 * ones(nS, 1);
%% ------------------------------------------------------------
% 1) Symbolic Setup
%% ------------------------------------------------------------
fprintf('Deriving Single-Point KKT System with %d Estimated Parameters...\n', nP-nX);
x_s   = sym('x_s',   [1 ns], 'real');
lam_s = sym('lam_s', [ns 1], 'real');
T_s   = sym('T_s',   'real');
yT_s  = sym('yT_s',  [1 5],  'real');
b_sym     = sym('b_sym', [nB 1], 'real');  
Eahat_sym = sym('Eahat', [nE 1], 'real'); 
lnA_sym   = sym('lnA_sym', [nA 1], 'real');
S_sym     = sym('S_sym', [nS 1], 'real');
p_s       = [b_sym; Eahat_sym; lnA_sym; S_sym];
% --- State Variables Assignment (11 Surface, 5 Gas) ---
th2 = x_s(1); thcoo = x_s(2); thcooh = x_s(3); th2cooh = x_s(4); 
th2co = x_s(5); toh = x_s(6); th3co = x_s(7); tch3oh = x_s(8); 
tco2 = x_s(9); tcooh = x_s(10); th2o = x_s(11);
ph2 = x_s(12); pco2 = x_s(13); ph2o = x_s(14); pco = x_s(15); pch3oh = x_s(16);
% Site Balance
ts = 1 - (th2 + thcoo + thcooh + th2cooh + th2co + toh + th3co + tch3oh + tco2 + tcooh + th2o); 
% --- Adsorption Constants (Using 5 Individual Estimated S_sym) ---
ka_h2  = S_sym(1) * sqrt(R * T_s / (2 * pi * 2));   
ka_co2 = S_sym(2) * sqrt(R * T_s / (2 * pi * 44));  
ka_h2o = S_sym(3) * sqrt(R * T_s / (2 * pi * 18));  
ka_co  = S_sym(4) * sqrt(R * T_s / (2 * pi * 28));  
ka_me  = S_sym(5) * sqrt(R * T_s / (2 * pi * 32));  
% --- Arrhenius Rate Constants (Estimated) ---
k_est = (exp(lnA_sym) .* sden) .* (T_s.^b_sym) .* exp(-(Eahat_sym * sym(Ea_scale)) / (sym(R) * T_s));
% --- 26 Unidirectional Rates ---
r1 = ka_h2 * ph2 * ts^2;      r2 = k_est(1) * th2^2;          
r3 = ka_co2 * pco2 * th2;     r4 = k_est(2) * thcoo;          
r5 = k_est(3) * thcoo * th2;  r6 = k_est(4) * thcooh * ts;    
r7 = k_est(5) * thcooh * th2; r8 = k_est(6) * th2cooh * ts;   
r9 = k_est(7) * th2cooh * ts; r10 = k_est(8) * th2co * toh;   
r11 = k_est(9) * toh * th2;   r12 = k_est(10) * th2o * ts;       
r13 = k_est(11) * th2co * th2; r14 = k_est(12) * th3co * ts;  
r15 = k_est(13) * th3co * th2; r16 = k_est(14) * tch3oh * ts; 
r17 = ka_co2 * pco2 * ts;     r18 = k_est(15) * tco2;         
r19 = k_est(16) * tco2 * th2; r20 = k_est(17) * tcooh * ts;   
r21 = k_est(18) * tcooh;      r22 = ka_co * pco * toh;        
r23 = k_est(19) * th2o;       r24 = ka_h2o * ph2o * ts;       
r25 = k_est(20) * tch3oh;     r26 = ka_me * pch3oh * ts;      
% --- ODE SYSTEM (h_s) ---
d_th2     = 2*(r1-r2) - (r3-r4) - (r5-r6) - (r7-r8) - (r11-r12) - (r13-r14) - (r15-r16) - (r19-r20);
d_thcoo   = (r3-r4) - (r5-r6);
d_thcooh  = (r5-r6) - (r7-r8);
d_th2cooh = (r7-r8) - (r9-r10);
d_th2co   = (r9-r10) - (r13-r14);
d_toh     = (r9-r10) - (r11-r12) + (r21-r22);
d_th3co   = (r13-r14) - (r15-r16);
d_tch3oh  = (r15-r16) - (r25-r26);
d_tco2    = (r17-r18) - (r19-r20);
d_tcooh   = (r19-r20) - (r21-r22);
d_th2o    = (r11-r12) - (r23-r24);
% Gas Phase ODEs
tau = .13; ph2_in = 143*3; pco2_in = 143; 
gas_scale = 1 / 3.41e8; 
d_ph2    = gas_scale * ((ph2_in - ph2)/tau) + (r2 - r1);
d_pco2   = gas_scale * ((pco2_in - pco2)/tau) + (r4 - r3) + (r18 - r17);
d_ph2o   = gas_scale * ((0 - ph2o)/tau) + (r23 - r24); 
d_pco    = gas_scale * ((0 - pco)/tau) + (r21 - r22);
d_pch3oh = gas_scale * ((0 - pch3oh)/tau) + (r25 - r26); 
h_s = [d_th2; d_thcoo; d_thcooh; d_th2cooh; d_th2co; d_toh; d_th3co; d_tch3oh; d_tco2; d_tcooh; d_th2o; ...
       d_ph2; d_pco2; d_ph2o; d_pco; d_pch3oh];
% Lagrangian and Objective
f_s = sym(0);
for i = 1:5 
    f_s = f_s + (x_s(i+11) - yT_s(i))^2;
end
f_s = f_s / sym(nc * 5); 
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
x0mat(:, 1:11)  = 0.05; 
x0mat(:, 12:16) = yTarget_Gas; 
p = [reshape(x0mat.',[],1); b_init; Eahat_init; lnA_init; S_init];
lamk = zeros(nH,1);
p = project_all_bounds_updated(p, nc, ns, nX, nB, nE, nA, nS, blo, bup, Eahatlo, Eahatup, lnAlo, lnAup, Slo, Sup);
p_best = p; sse_best = inf; h_best = inf; r_best = inf;
best_h_vec = zeros(nH, 1); iter_best = 0;
%% ------------------------------------------------------------
% 3) PHASE 1: Full Newton Solver
%% ------------------------------------------------------------
fprintf('\n=== Full Newton Log-Scaled Solve ===\n');
maxIter = 2500; last_maxr = inf; stag_count = 0;
for k = 1:maxIter
    [rk, Jk, hk, sse_total] = build_full_system(p, lamk, Tvec, yTarget_Gas, funcs, nc, ns, nX, nP, nH);
    maxr = max(abs(rk)); maxh = max(abs(hk));
    curr_sse = sse_total * num_pts;
    
    if maxh < 5e-4 && maxr < 5e-2 && curr_sse < 1e-6
        sse_best = curr_sse; h_best = maxh; r_best = maxr;
        p_best = p; best_h_vec = hk; iter_best = k;
        if maxh <= 2e-5 && maxr < 5e-5, break; end
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
            dx = -1.0 * lsqminnorm(Jh_x, hk_full);
            p(1:nX) = p(1:nX) + dx;
            p = project_all_bounds_updated(p, nc, ns, nX, nB, nE, nA, nS, blo, bup, Eahatlo, Eahatup, lnAlo, lnAup, Slo, Sup);
            [~, ~, hk_chk, ~] = build_full_system(p, lamk, Tvec, yTarget_Gas, funcs, nc, ns, nX, nP, nH);
            if max(abs(hk_chk)) < 1e-6, break; end 
        end
    end
    
    dz = lsqminnorm(Jk, rk);
    alpha_try = 1.0; rho = 0.5; Phi0 = 0.5*(rk.'*rk); accepted = false;
    for bt = 1:30
        zk_try = [p; lamk] - alpha_try*dz;
        p_try = project_all_bounds_updated(zk_try(1:nP), nc, ns, nX, nB, nE, nA, nS, blo, bup, Eahatlo, Eahatup, lnAlo, lnAup, Slo, Sup);
        lam_try = zk_try(nP+1:end);
        [r_try, ~, ~, ~] = build_full_system(p_try, lam_try, Tvec, yTarget_Gas, funcs, nc, ns, nX, nP, nH);
        if 0.5*(r_try.'*r_try) <= Phi0, p = p_try; lamk = lam_try; accepted = true; break; end
        alpha_try = alpha_try * rho;
    end
    if ~accepted, lamk = 0.8*lamk; end
    if mod(k,100) == 0 || k == 1, fprintf('%5d | R:%10.2e H:%10.2e SSE:%10.2e\n', k, maxr, maxh, curr_sse); end
end
%% ------------------------------------------------------------
% 4) Final Report
%% ------------------------------------------------------------
p = p_best; 
fprintf('\n================ FINAL BEST RESULTS =================\n');
Xfinal = reshape(p(1:nX), [ns nc]).';
b_opt     = p(nX+1 : nX+nB);
Eahat_opt = p(nX+nB+1 : nX+nB+nE);
lnA_opt   = p(nX+nB+nE+1 : nX+nB+nE+nA);
S_opt     = p(nX+nB+nE+nA+1 : nP);
Ea_opt    = Eahat_opt * Ea_scale;
A_opt     = exp(lnA_opt);
fprintf('Iteration where best value was observed: %d\n', iter_best);
fprintf('Lowest Total SSE Found: %.4e\n', sse_best);
fprintf('Overall Max Absolute Constraint Violation (h_max): %.2e\n', h_best);
fprintf('Max KKT Residual (r_max) at Best: %.2e\n\n', r_best);
fprintf('--- Optimized Adsorption (Sticking Coefficients) ---\n');
fprintf('S_H2:    %.4f\n', S_opt(1));
fprintf('S_CO2:   %.4f\n', S_opt(2));
fprintf('S_H2O:   %.4f\n', S_opt(3));
fprintf('S_CO:    %.4f\n', S_opt(4));
fprintf('S_CH3OH: %.4f\n\n', S_opt(5));
fprintf('--- Max Absolute Constraint Violation Per Equation ---\n');
h_mat = reshape(best_h_vec, [ns, nc]).';
max_h_per_eq = max(abs(h_mat), [], 1).';
EqNames = {'dy1 (th2)'; 'dy2 (thcoo)'; 'dy3 (thcooh)'; 'dy4 (th2cooh)'; 'dy5 (th2co)'; 'dy6 (toh)'; ...
           'dy7 (th3co)'; 'dy8 (tch3oh)'; 'dy9 (tco2)'; 'dy10 (tcooh)'; 'dy11 (th2o)'; ...
           'dy12 (ph2)'; 'dy13 (pco2)'; 'dy14 (ph2o)'; 'dy15 (pco)'; 'dy16 (pch3oh)'};
MaxViolationTable = table(EqNames, max_h_per_eq, 'VariableNames', {'Equation', 'Max_Abs_Violation'});
disp(MaxViolationTable);
fprintf('\n--- Final Estimated Kinetics (A, b, Ea) ---\n');
StepNames = arrayfun(@(x) sprintf('k_est%d', x), (1:nr_est)', 'UniformOutput', false);
ParamTable = table(StepNames, A_opt, b_opt, Ea_opt, ...
    'VariableNames', {'Rate','A_Optimized','b','Ea_J_mol'});
disp(ParamTable);
fprintf('\n--- Final Surface Coverages (y1 - y11) ---\n');
RowNamesT = arrayfun(@(t) sprintf('%.1f K', t), Tvec, 'UniformOutput', false);
SurfaceTable = array2table(Xfinal(:, 1:11), ...
    'VariableNames', {'th2','thcoo','thcooh','th2cooh','th2co','toh','th3co','tch3oh','tco2','tcooh','th2o'}, ...
    'RowNames', RowNamesT);
disp(SurfaceTable);
fprintf('\n--- Final Gas Pressures/Conc (y12 - y16) ---\n');
GasTable = array2table(Xfinal(:, 12:16), ...
    'VariableNames', {'pH2','pCO2','pH2O','pCO','pCH3OH'}, ...
    'RowNames', RowNamesT);
disp(GasTable);
toc;
%% ------------------------------------------------------------
% 6) Visual Confirmation
%% ------------------------------------------------------------
figure('Color', 'w', 'Position', [100, 100, 1200, 400]);
species_names = {'H_2', 'CO_2', 'H_2O', 'CO', 'CH_3OH'};
colors = lines(5);
for i = 1:5
    subplot(1, 5, i);
    plot(Tvec, yTarget_Gas(:, i), 'o', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', 'DisplayName', 'Data');
    hold on;
    plot(Tvec, Xfinal(:, i+11), '-', 'LineWidth', 2, 'Color', colors(i,:), 'DisplayName', 'Model');
    title(species_names{i});
    xlabel('Temperature (K)');
    grid on;
    if i == 5, legend('Location', 'best'); end
end
sgtitle('Methanol System: Model vs. Data');
%% ------------------------------------------------------------
% 5) LOCAL FUNCTIONS
%% ------------------------------------------------------------
function [R, J, h_all, sse_total] = build_full_system(p, lam, Tvec, yTarget, funcs, nc, ns, nX, nP, nH)
    R = zeros(nP + nH, 1);
    J = spalloc(nP + nH, nP + nH, (nP+nH)*20); 
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
function p = project_all_bounds_updated(p, nc, ns, nX, nB, nE, nA, nS, blo, bup, Eahatlo, Eahatup, lnAlo, lnAup, Slo, Sup)
    X = reshape(p(1:nX), [ns nc]).'; 
    for c = 1:nc
        X(c, 12:16) = min(max(X(c, 12:16), 0.0), 1000.0);
        surf = min(max(X(c, 1:11), 0.0), 1.0);
        if sum(surf) > 1.0, surf = surf/sum(surf); end
        X(c, 1:11) = surf;
    end
    p(1:nX) = reshape(X.', [], 1);
    p(nX+1 : nX+nB) = min(max(p(nX+1 : nX+nB), blo), bup);
    p(nX+nB+1 : nX+nB+nE) = min(max(p(nX+nB+1 : nX+nB+nE), Eahatlo), Eahatup);
    p(nX+nB+nE+1 : nX+nB+nE+nA) = min(max(p(nX+nB+nE+1 : nX+nB+nE+nA), lnAlo), lnAup);
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