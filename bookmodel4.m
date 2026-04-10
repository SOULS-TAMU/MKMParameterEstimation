%% MKM: SYMBOLIC Full-Newton + Log-Scaling + Adaptive Kick
% Added: Best-Result Tracker (Lowest SSE within Feasibility & Optimality)
% Update: Reverted to A, b, Ea optimization with Individual Targeted Bounds
% Update: Ea Parameter Scaling added
% Update: Objective function (SSE) now ONLY targets gas species (x1-x3)
% Update: Strict maxr < 1e-4 requirement added to best-result tracker
clear; clc; tic;
rng(42); % Locks the random seed for 100% consistent results across runs

%% ------------------------------------------------------------
% 0) Constants & Data
%% ------------------------------------------------------------
R     = 8.314462618;     
alpha = 1e4;             
Tvec  = [300; 350; 400; 450; 500];
nc    = numel(Tvec); ns = 7; nr = 4;
nX    = nc*ns; nA = 2*nr; nB = 2*nr; nE = 2*nr;
nP    = nX + nA + nB + nE; nH = nc*7;

% Full Target Data (Gas x1-x3 and Surface x4-x7) extracted from images
xTarget = [ ...
11.32778950, 20.66389470, 8.67221052, 0.45485613, 0.10003515, 6.86E-06,   0.44510185;
10.21865450, 20.10932730, 9.78134549, 0.49225711, 0.12901088, 4.40E-05,   0.37868804;
9.31861299,  19.65930650, 10.6813870, 0.51142081, 0.15499735, 0.00017460, 0.33340723;
8.62075545,  19.31037770, 11.3792445, 0.52109647, 0.17788792, 0.00050499, 0.30051062;
8.07111327,  19.03555660, 11.9288867, 0.52527889, 0.19793825, 0.00117295, 0.27560991 ];

% --- GROUND TRUTH PARAMETERS (From Image) ---
A_gt  = 1e3 * ones(8,1);
b_gt  = 1   * ones(8,1);
Ea_gt = [10000; 4000; 20000; 5000; 30000; 6000; 40000; 7000];

% SCALING FACTOR FOR Ea
Ea_scale = 10000;
Eahat_gt = Ea_gt / Ea_scale;

% --- INDIVIDUAL BOUNDS ---
lnAlo = log(A_gt) - 2;   
lnAup = log(A_gt) + 2;
blo   = b_gt - 1.5;      
bup   = b_gt + 1.5;
Eahatlo = max(0, Eahat_gt - 0.5); 
Eahatup = Eahat_gt + 0.5;

%% ------------------------------------------------------------
% 1) Symbolic Setup
%% ------------------------------------------------------------
fprintf('Deriving KKT System with Log-A Scaling and Ea Scaling...\n');
x     = sym('x',     [nc ns], 'real');
lnA   = sym('lnA',   [nA 1],  'real');  
b     = sym('b',     [nB 1],  'real');
Eahat = sym('Eahat', [nE 1],  'real'); 
pvec  = [reshape(x.', [], 1); lnA; b; Eahat];

% Objective f: SSE scaled by total points (ONLY Gas Species x1-x3)
f = sym(0);
num_pts = nc * 3; 
for c = 1:nc
    for i = 1:3 
        f = f + (x(c,i) - sym(xTarget(c,i)))^2;
    end
end
f = f / sym(num_pts); 

h = sym(zeros(nH,1));
for c = 1:nc
    T = sym(Tvec(c)); x_c = x(c,:);
    ks = exp(lnA) .* exp(b*log(T) - (Eahat*sym(Ea_scale))/(sym(R)*T));
    
    dx1 = (0-x_c(1)) + sym(alpha)*(-(ks(1)*x_c(1)*x_c(7)) + (ks(2)*x_c(4)));
    dx2 = (15-x_c(2)) + sym(alpha)*(-(ks(3)*x_c(2)*x_c(7)^2) + (ks(4)*x_c(5)^2));
    dx3 = (20-x_c(3)) + sym(alpha)*(-(ks(7)*x_c(3)*x_c(7)) + (ks(8)*x_c(6)));
    dx4 = (ks(1)*x_c(1)*x_c(7)-ks(2)*x_c(4)) - (ks(5)*x_c(4)*x_c(5)-ks(6)*x_c(6)*x_c(7));
    dx5 = 2*(ks(3)*x_c(2)*x_c(7)^2-ks(4)*x_c(5)^2) - (ks(5)*x_c(4)*x_c(5)-ks(6)*x_c(6)*x_c(7));
    dx6 = (ks(5)*x_c(4)*x_c(5)-ks(6)*x_c(6)*x_c(7)) + (ks(7)*x_c(3)*x_c(7)-ks(8)*x_c(6));
    mbeq = sum(x_c(4:7)) - 1;
    h((c-1)*7+1:c*7) = [dx1; dx2; dx3; dx4; dx5; dx6; mbeq];
end

lam = sym('lam', [nH 1], 'real');
L = f + lam.'*h;
rsym = [gradient(L, pvec); h];
zsym = [pvec; lam];
Jfun = matlabFunction(jacobian(rsym, zsym), 'Vars', {pvec, lam});
rfun = matlabFunction(rsym, 'Vars', {pvec, lam});
hfun = matlabFunction(h, 'Vars', {pvec});
Jhfun = matlabFunction(jacobian(h, pvec), 'Vars', {pvec});
ffun = matlabFunction(f, 'Vars', {pvec});

%% ------------------------------------------------------------
% 2) Initialization
%% ------------------------------------------------------------
x0mat = xTarget; 

p0_lnA   = log(A_gt) + 0.1;
p0_b     = b_gt + 0.1;
p0_Eahat = Eahat_gt + 0.05; 
p = [reshape(x0mat.',[],1); p0_lnA; p0_b; p0_Eahat];
lamk = zeros(nH,1);
p = project_all_bounds(p, nc, ns, nX, nA, nB, nE, lnAlo, lnAup, blo, bup, Eahatlo, Eahatup);

% Best Tracker Memory
p_best = p;
sse_best = inf;
h_best = inf;
r_best = inf;
best_h_vec = zeros(nH, 1);
iter_best = 0;

%% ------------------------------------------------------------
% 3) PHASE 1: Full Newton Solver
%% ------------------------------------------------------------
fprintf('\n=== PHASE 1: Full Newton Log-Scaled Solve ===\n');
maxIter = 20000; 
last_maxr = inf; stag_count = 0;

for k = 1:maxIter
    rk = rfun(p, lamk); Jk = Jfun(p, lamk); hk = hfun(p);
    maxr = max(abs(rk)); maxh = max(abs(hk));
    curr_sse = ffun(p) * num_pts;
    
    % Save the best result (Strict Criteria: Feasible, Optimal, Lowest SSE)
    % if maxh < 1e-3 && maxr < 1e-4 && curr_sse < sse_best
    %     sse_best = curr_sse;
    %     h_best = maxh;
    %     r_best = maxr;
    %     p_best = p;
    %     best_h_vec = hk;
    %     iter_best = k;
    % end


% Define your "Satisfaction" Thresholds
sse_target = 1e-6; % What you consider "Zero" for SSE
h_target = 9e-5;   % Your required physics feasibility

if maxh < 9e-5 && maxr < 9e-5 && curr_sse < sse_best
    % 1) Update the "Best" records
    sse_best = curr_sse;
    h_best = maxh;
    r_best = maxr;
    p_best = p;
    best_h_vec = hk;
    iter_best = k;
    
    % 2) THE BREAK CONDITION
    % If the current "Best" is also "Good Enough" based on targets, stop now.
    if curr_sse <= sse_target && maxh <= h_target
        fprintf('\n[CONVERGED] Target SSE and Feasibility met at iteration %d. Breaking...\n', k);
        break; 
    end
end
    
    if abs(maxr - last_maxr) < 1e-12, stag_count = stag_count + 1;
    else, stag_count = 0; end
    last_maxr = maxr;
    
    if stag_count > 25
        fprintf(' [KICK] ');
        p = p + (1e-4 * randn(size(p))); 
        lamk = 0.5 * lamk; stag_count = 0;
    end
    
    if maxh > 1e-3
        for kr = 1:15
            p = p - 0.5 * lsqminnorm(Jhfun(p), hfun(p));
            p = project_all_bounds(p, nc, ns, nX, nA, nB, nE, lnAlo, lnAup, blo, bup, Eahatlo, Eahatup);
            if max(abs(hfun(p))) < 1e-8, break; end
        end
    end
    
    dz = lsqminnorm(Jk, rk);
    alpha_try = 1.0; rho = 0.5; c1 = 1e-4; Phi0 = 0.5*(rk.'*rk);
    accepted = false;
    for bt = 1:30
        zk_try = [p; lamk] - alpha_try*dz;
        p_try = project_all_bounds(zk_try(1:nP), nc, ns, nX, nA, nB, nE, lnAlo, lnAup, blo, bup, Eahatlo, Eahatup);
        lam_try = zk_try(nP+1:end);
        r_try = rfun(p_try, lam_try);
        if 0.5*(r_try.'*r_try) <= Phi0*(1 - c1*alpha_try)
            p = p_try; lamk = lam_try; alpha_used = alpha_try; accepted = true; break;
        end
        alpha_try = alpha_try * rho;
    end
    
    if ~accepted, lamk = 0.8*lamk; alpha_used = 0; end
    if mod(k,100) == 0 || k == 1
        fprintf('%5d | R:%10.2e H:%10.2e SSE:%10.4f\n', k, maxr, maxh, curr_sse);
    end
end

%% ------------------------------------------------------------
% 4) Final Report (Using Best Values Saved)
%% ------------------------------------------------------------
p = p_best; % Load best snapshot
fprintf('\n================ FINAL BEST RESULTS =================\n');
Xfinal = reshape(p(1:nX), [ns nc]).';
lnA_opt   = p(nX+(1:nA)); 
b_opt     = p(nX+nA+(1:nB)); 
Eahat_opt = p(nX+nA+nB+(1:nE));

% Unscale Ea for reporting
Ea_opt = Eahat_opt * Ea_scale;

fprintf('Iteration where best value was observed: %d\n', iter_best);
fprintf('Lowest Total SSE Found (Gas Species 1-3 only): %.6f\n', sse_best);
fprintf('Max Absolute Constraint Violation (h_max): %.2e\n', h_best);
fprintf('Max KKT Residual (r_max) at Best: %.2e\n\n', r_best);

fprintf('--- Detailed Constraint Violations at Best (h_i) ---\n');
h_mat = reshape(best_h_vec, [7, nc]).';
disp(array2table(h_mat, 'VariableNames', {'dx1','dx2','dx3','dx4','dx5','dx6','SiteBal'}, ...
    'RowNames', arrayfun(@(t) sprintf('T_%dK', t), Tvec, 'UniformOutput', false)));

fprintf('--- Final Concentrations (x1-x7) ---\n');
disp(array2table(Xfinal, 'VariableNames', {'x1','x2','x3','x4','x5','x6','x7'}, ...
    'RowNames', arrayfun(@(t) sprintf('T_%dK', t), Tvec, 'UniformOutput', false)));

fprintf('--- Final Estimated Parameters (A, b, Ea) ---\n');
StepNames = {'f1'; 'b1'; 'f2'; 'b2'; 'f3'; 'b3'; 'f4'; 'b4'};
ParamTable = table(StepNames, exp(lnA_opt), lnA_opt, b_opt, Ea_opt, ...
    'VariableNames', {'Step','A','lnA','b','Ea_Jmol'});
disp(ParamTable);

toc;

%% ------------------------------------------------------------
% 5) LOCAL FUNCTIONS
%% ------------------------------------------------------------
function p = project_all_bounds(p, nc, ns, nX, nA, nB, nE, lnAlo, lnAup, blo, bup, Eahatlo, Eahatup)
    x_states = p(1:nX); X = reshape(x_states, [ns nc]).'; 
    for c = 1:nc, X(c,4:7) = proj_capped_simplex_bisect(X(c,4:7).', 1.0, 0.0, 1.0).'; end
    p(1:nX) = reshape(X.', [], 1);
    
    iA = nX + (1:nA); iB = nX+nA + (1:nB); iE = nX+nA+nB + (1:nE);
    p(iA) = min(max(p(iA), lnAlo), lnAup); 
    p(iB) = min(max(p(iB), blo), bup); 
    p(iE) = min(max(p(iE), Eahatlo), Eahatup);
end

function x = proj_capped_simplex_bisect(y, targetSum, lb, ub)
    tau_lo = min(y)-ub-1; tau_hi = max(y)-lb+1;
    for it = 1:50
        tau = 0.5*(tau_lo + tau_hi); x = min(max(y - tau, lb), ub);
        if sum(x) > targetSum, tau_lo = tau; else, tau_hi = tau; end
    end
    x = min(max(y - tau_hi, lb), ub);
end