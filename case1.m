%% Feasibility-first + your original KKT/FB residual solver (with backtracking)
clear; clc;
tic;
%% -------------------------
% 1) Symbolic variables
% -------------------------
syms x1 x2 K real
x = [x1; x2; K];

syms lam1 lam2 real
lam = [lam1; lam2];

syms mu1 real
mu = [mu1];

syms epsFB real

%% -------------------------
% 2) Define objective and constraints
% -------------------------
f  = (0.081 - x1)^2 + (47.6 - x2)^2;

h1 = 15*2.5 - 15*x1 - K*x1*x2*1250;
h2 = 15*50  - 15*x2 - K*x1*x2*1250;
h  = [h1; h2];

g  = [0];

%% -------------------------
% 3) Lagrangian and residual r(z)
% -------------------------
L  = f + lam.'*h + mu.'*g;
rL = gradient(L, x);                 % 3x1

s     = sqrt(mu.^2 + (-g).^2 + epsFB^2);
phiFB = s - mu + g;                  % 1x1

rsym = simplify([rL; h; phiFB]);     % 3 + 2 + 1 = 6x1

%% -------------------------
% 4) Jacobians
% -------------------------
zsym = [x; lam; mu];                    % 3 + 2 + 1 = 6x1
Jsym = simplify(jacobian(rsym, zsym));  % 6x6

% Feasibility-only Jacobian: Jh = dh/dx (2x3)
Jhsym = simplify(jacobian(h, x));       % 2x3

%% -------------------------
% 5) Numeric evaluators
% -------------------------
Jfun  = matlabFunction(Jsym,  'Vars', {x, lam, mu, epsFB});
rfun  = matlabFunction(rsym,  'Vars', {x, lam, mu, epsFB});
ffun  = matlabFunction(f,     'Vars', {x});

hfun  = matlabFunction(h,     'Vars', {x});
Jhfun = matlabFunction(Jhsym, 'Vars', {x});

%% -------------------------
% 6) Settings
% -------------------------
epsval = 1e-2;
alpha  = 1;

maxIter = 10000;
tol_comp = 1e-12;

% Backtracking (for both phases)
c1   = 1e-4;
rho  = 0.5;
amin = 1e-14;
maxBT = 40;

%% -------------------------
% 6.5) FEASIBILITY-FIRST settings (NEW)
% -------------------------
do_feasibility_first = true;
maxFeasIter = 150;          % 50–200 is typical
tol_h = 1e-15;               % stop feasibility phase when max|h| small
alpha_feas0 = 1.0;          % initial step for feasibility phase

%% -------------------------
% 6.6) Initial guess (same idea as yours)
% -------------------------
xk   = [1; 1; 1];
lamk = [1; 1];
muk  = [1];

%% ============================================================
% PHASE 0: Feasibility-first (drive h(x) -> 0)
% ============================================================
if do_feasibility_first
    fprintf('\n=== PHASE 0: Feasibility-first (minimize 0.5*||h(x)||^2) ===\n');
    fprintf('Iter | max|h|        ||hx||2        ||dx||2       alpha_used\n');
    fprintf('------------------------------------------------------------\n');

    for kf = 1:maxFeasIter
        hk  = hfun(xk);          % 2x1
        Jh  = Jhfun(xk);         % 2x3

        maxh = max(abs(hk));
        nh   = norm(hk,2);

        if maxh <= tol_h
            fprintf('Feasibility reached: max|h| <= %.1e at iter %d\n', tol_h, kf);
            break;
        end

        % Gauss–Newton step for feasibility: solve Jh*dx ≈ hk  => dx = lsqminnorm(Jh, hk)
        dx = lsqminnorm(Jh, hk);
        dxn = norm(dx,2);

        % Merit for feasibility
        Phi0 = 0.5*(hk.'*hk);

        % Backtracking on feasibility merit
        alpha_try = alpha_feas0;
        alpha_used = alpha_try;

        for bt = 1:maxBT
            x_try = xk - alpha_try*dx;

            h_try = hfun(x_try);
            Phi_try = 0.5*(h_try.'*h_try);

            if Phi_try <= Phi0*(1 - c1*alpha_try)
                alpha_used = alpha_try;
                xk = x_try;
                break;
            end

            alpha_try = rho*alpha_try;
            if alpha_try < amin
                alpha_used = alpha_try;
                xk = xk - alpha_used*dx; % tiny step fallback
                break;
            end
        end

        fprintf('%4d | %12.4e  %12.4e  %12.4e  %10.3e\n', ...
            kf, maxh, nh, dxn, alpha_used);
    end

    % Optional: reset multipliers after feasibility phase
    lamk = [0; 0];
    muk  = [1];  % keep your mu as-is
end


%% -------- DROP-IN: Log values AFTER feasibility-first (effective Phase 1 start) --------
x_init1   = xk;
lam_init1 = lamk;
mu_init1  = muk;




%% ============================================================
% PHASE 1: Your original full KKT/FB residual solver
% ============================================================
fprintf('\n=== PHASE 1: Full residual solve ===\n');
fprintf('Iter | max|r|        min|r|        ||delz||2     alpha_used    stop?\n');
fprintf('-----------------------------------------------------------------------\n');
tol_break=2e-8;
for k = 1:maxIter
    rk = rfun(xk, lamk, muk, epsval);   % 6x1
    Jk = Jfun(xk, lamk, muk, epsval);   % 6x6

    stop_now = all(abs(rk) <= tol_comp);

    if max(abs(rk)) < tol_break
        fprintf('Target tolerance %.1e reached at iter %d.\n', tol_break, k);
        break; 
    end

    % Your step (same style)
    delz = lsqminnorm(Jk, rk);
    delz2 = norm(delz,2);

    % Merit on full residual
    Phi0 = 0.5*(rk.'*rk);

    % Backtracking (same as earlier message)
    zk  = [xk; lamk; muk];
    p   = -delz;

    alpha_try = alpha;
    alpha_used = alpha_try;
    accepted = false;

    for bt = 1:maxBT
        zk_try = zk + alpha_try*p;

        x_try   = zk_try(1:3);
        lam_try = zk_try(4:5);
        mu_try  = zk_try(6);

        r_try = rfun(x_try, lam_try, mu_try, epsval);
        Phi_try = 0.5*(r_try.'*r_try);

        if Phi_try <= Phi0*(1 - c1*alpha_try)
            accepted = true;
            alpha_used = alpha_try;
            zk1 = zk_try;
            break;
        end

        alpha_try = rho*alpha_try;
        if alpha_try < amin
            break;
        end
    end

    if ~accepted
        alpha_used = alpha_try;
        zk1 = zk + alpha_used*p;   % tiny step fallback
    end

    fprintf('%4d | %12.4e  %12.4e  %12.4e   %10.3e   %d\n', ...
        k, max(abs(rk)), min(abs(rk)), delz2, alpha_used, stop_now);

    if stop_now
        fprintf('Stopping: all |r_i| <= %.1e\n', tol_comp);
        break;
    end

    % Unpack (same)
    xk   = zk1(1:3);
    lamk = zk1(4:5);
    muk  = zk1(6);
end

%% -------------------------
% 8) Final report
% -------------------------
r_final = rfun(xk, lamk, muk, epsval);

fprintf('\n================ FINAL =================\n');
fprintf('x*   = [%.10f %.10f %.10f]^T\n', xk);
fprintf('lam* = [%.10f %.10f]^T\n', lamk);
fprintf('mu*  = [%.10f]^T\n', muk);
fprintf('f(x*) = %.12e\n', ffun(xk));

fprintf('h(x*) = \n');
disp(hfun(xk));

fprintf('r(z*) = \n');
disp(r_final);

fprintf('max(abs(r)) = %.3e\n', max(abs(r_final)));
fprintf('All components <= tol_comp? %d\n', all(abs(r_final) <= tol_comp));


fprintf('\n===== VALUES USED TO START PHASE 1 (after feasibility-first) =====\n');
fprintf('x_init1   = [%.16g  %.16g  %.16g]^T\n', x_init1(1), x_init1(2), x_init1(3));
fprintf('lam_init1 = [%.16g  %.16g]^T\n',      lam_init1(1), lam_init1(2));
fprintf('mu_init1  = [%.16g]^T\n',             mu_init1(1));

h1v = hfun(x_init1);
fprintf('h(x_init1) = [%.16g  %.16g]^T\n', h1v(1), h1v(2));
fprintf('max|h(x_init1)| = %.6e\n', max(abs(h1v)));
fprintf('==================================================================\n');

toc;