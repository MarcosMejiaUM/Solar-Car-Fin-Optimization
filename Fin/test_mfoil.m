%-------------------------------------------------------------------------------
% test_mfoil.m  —  Step 1 of MFoil integration: verify MFoil runs on this
% machine and its gradients are intact. Pure MATLAB, no Python bridge.
%
% HOW TO RUN (Windows MATLAB):
%   1. Put this file in the SAME folder as mfoil.m
%   2. cd to that folder in MATLAB
%   3. >> test_mfoil
%   4. Copy the entire Command Window output back.
%
% WHAT IT CHECKS:
%   A) NACA 0012 polar at Re=6e5 -- compare cl/cd to our XFOIL D1 baseline
%      (XFOIL gave: lift slope ~0.105/deg, so cl(a)~0.105*a; CDmin ~0.0059).
%      If MFoil agrees within a few %, the two solvers are consistent.
%   B) ping() -- finite-difference verification of the derivative building
%      blocks behind the adjoint. Rates printed should be ~2 (2nd order).
%      Small errors / rates near 2 = the gradient machinery is sound here.
%-------------------------------------------------------------------------------
clear; clc;
fprintf('================================================================\n');
fprintf(' MFoil verification (Step 1) — version check + polar + ping\n');
fprintf('================================================================\n');

Re = 6e5;

% ---- A) NACA 0012 polar, compare to XFOIL ----
fprintf('\n(A) NACA 0012 @ Re = %.0e  (compare to XFOIL: slope~0.105/deg, CDmin~0.0059)\n', Re);
fprintf(' alpha      cl         cd         cm\n');
alphas = [0 2 4 6 8];
cls = zeros(size(alphas)); cds = zeros(size(alphas));
for i = 1:numel(alphas)
    a = alphas(i);
    m = mfoil('naca','0012','npanel',199);
    m.param.verb   = 0;        % quiet
    m.param.doplot = false;    % no figures
    m.setoper('alpha',a,'Re',Re,'Ma',0,'visc',true);
    m.solve();
    cls(i) = m.post.cl; cds(i) = m.post.cd;
    fprintf('%6.1f   %8.4f   %8.5f   %8.4f\n', a, m.post.cl, m.post.cd, m.post.cm);
end
% crude lift slope from a=0..4
slope = (cls(3) - cls(1)) / (alphas(3) - alphas(1));
fprintf('  --> lift slope (0..4 deg) = %.4f /deg   (XFOIL ~0.105)\n', slope);
fprintf('  --> min cd                = %.5f        (XFOIL ~0.0059)\n', min(cds));

% ---- B) ping(): derivative / adjoint verification ----
fprintf('\n(B) ping() derivative verification (rates should be ~2.0):\n');
fprintf('----------------------------------------------------------------\n');
m = mfoil('naca','0012','npanel',199);
m.param.verb   = 0;
m.param.doplot = false;
m.ping();
fprintf('----------------------------------------------------------------\n');
fprintf('\nDone. Copy everything above back.\n');