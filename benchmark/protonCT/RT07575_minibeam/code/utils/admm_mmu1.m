function [weights, info, history] = admm_mmu1(ip, var_CG)
%ADMM_MMU1 Physical-dose active-set ADMM with nonnegative spot weights.
% The supplied single-precision CG MEX solves each quadratic subproblem.
% Return the feasible projected variable z, rather than an unclipped primal x.
nX = var_CG.var_AtA.nX;
x = zeros(nX,1,'single');
if isfield(ip,'x0')
    x = single(ip.x0(:));
    assert(numel(x) == nX && all(isfinite(x)), 'Invalid ADMM warm start.');
end
z = max(x,0);
u = zeros(nX,1,'single');
rho = single(ip.mup);
var_CG.mu_xs = rho;
assert(ip.mu_min == 0, 'This physical optimizer supports nonnegative weights only.');
assert(isfinite(rho) && rho > 0, 'ADMM penalty must be positive.');
history = struct('iteration',{},'objective',{},'D95_Gy',{}, ...
    'primal_residual',{},'dual_residual',{},'primal_tolerance',{}, ...
    'dual_tolerance',{},'cg_relative_residual_squared',{},'cg_iterations',{});
converged = false;
stable = 0;
previousObjective = inf;
for iteration = 1:ip.N_iter
    [rhs,var_CG.var_AtA] = update_ac(x,var_CG.var_AtA);
    b = var_CG.AtmX(rhs,var_CG.var_AtA);
    [x,cgError,cgIterations] = CG033114(x, b+rho*(z-u), ...
        var_CG.JtJ,var_CG,var_CG.cg_tol,var_CG.cg_iter);
    assert(all(isfinite(x)) && isfinite(cgError), 'CG returned invalid spot weights.');
    zOld = z;
    z = max(x+u,0);
    u = u+x-z;
    primal = norm(double(x)-double(z));
    dual = double(rho)*norm(double(z)-double(zOld));
    primalTolerance = sqrt(nX)*ip.absolute_tol + ...
        ip.relative_tol*max(norm(double(x)),norm(double(z)));
    dualTolerance = sqrt(nX)*ip.absolute_tol*double(rho) + ...
        ip.relative_tol*norm(double(rho*u));
    % Evaluate every objective at the feasible plan, with updated active sets.
    [~, evaluation] = update_ac(z,var_CG.var_AtA);
    [obj, doses] = calc_obj_dvh(z,evaluation);
    objective = sum(obj(:));
    target = sort(doses{1},'descend');
    D95 = target(max(1,round(numel(target)*0.95)));
    relativeChange = abs(objective-previousObjective)/max(1,abs(previousObjective));
    if iteration >= ip.minimum_iter && primal <= primalTolerance && ...
            dual <= dualTolerance && relativeChange <= ip.objective_tol
        stable = stable+1;
    else
        stable = 0;
    end
    history(iteration) = struct('iteration',iteration,'objective',objective, ...
        'D95_Gy',D95,'primal_residual',primal,'dual_residual',dual, ...
        'primal_tolerance',primalTolerance,'dual_tolerance',dualTolerance, ...
        'cg_relative_residual_squared',double(cgError),'cg_iterations',double(cgIterations));
    fprintf('ADMM %3d: obj %.7g, D95 %.5g Gy, r %.3g/%.3g, s %.3g/%.3g, CG %d %.3g\n', ...
        iteration,objective,D95,primal,primalTolerance,dual,dualTolerance,cgIterations,cgError);
    previousObjective = objective;
    if stable >= 3
        converged = true;
        break;
    end
end
weights = z;
info = struct('converged',converged,'iterations',iteration, ...
    'solver','ADMM + CG033114/JtJ_id MEX', ...
    'cg_tolerance_convention','squared residual / squared initial residual', ...
    'penalty',double(rho),'last_objective',objective);
if ~converged
    warning('ADMM:IterationLimit','Reached %d ADMM iterations before convergence.',iteration);
end
end
