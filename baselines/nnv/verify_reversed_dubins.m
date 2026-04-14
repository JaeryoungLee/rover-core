% verify_reversed_dubins.m
%
% NNV v2.0 baseline: Backward Reachable Tube (BRT) via forward
% reachability with time-reversed Dubins car dynamics.
%
% Usage:
%   verify_reversed_dubins          % default ntheta=100
%   ntheta_arg=50; verify_reversed_dubins
%
% Uses NNV's canonical pipeline:
%   - approx-star for NN controller reachability
%   - CORA zonotope for nonlinear plant dynamics
%
% System: Dubins car, v=1.0 m/s, reversed dynamics
% Controller: ReLU MLP, 3->128->128->1

fprintf('=== NNV v2.0: Reversed Dubins BRT ===\n');

% -------------------------------------------------------------------------
% 1. Load NN controller
% -------------------------------------------------------------------------
fprintf('Loading NN ...\n');
data = load('../../baselines/nn/RoverBaseline_MPC_NN.mat');
n = data.num_layers;
Layers = {};
for i = 1:(n-1)
    Layers{end+1} = LayerS(data.weights{i}, data.bias{i}, 'poslin');
end
Layers{end+1} = LayerS(data.weights{n}, data.bias{n}, 'purelin');
Controller = NN(Layers, [], data.input_dim, data.output_dim, 'dubins_ctrl');

% -------------------------------------------------------------------------
% 2. Create CORA plant
% -------------------------------------------------------------------------
fprintf('Creating CORA plant ...\n');
dynamics = @(x, u) [-cos(x(3)); -sin(x(3)); -u(1)];
controlPeriod = 0.1;
reachStep     = 0.01;
% T and numCtrlSteps set in config section below
plant = NonLinearODE(3, 1, dynamics, reachStep, controlPeriod, eye(3));

% Warm up CORA tensors
plant.stepReachStar(Star([15.5; -0.5; 0.0], [16.0; 0.0; 0.1]), Star(0, 0));
fprintf('  CORA ready.\n');

% -------------------------------------------------------------------------
% 3. Configuration
% -------------------------------------------------------------------------
obs_lb = [15; -1; -pi];
obs_ub = [17;  1;  pi];
epsilon = [0.5; 0.5; 0.1];

nx = 1; ny = 1;
if ~exist('ntheta_arg', 'var'), ntheta_arg = 100; end
if ~exist('T_arg', 'var'), T_arg = 1.0; end
ntheta = ntheta_arg;
T = T_arg;
numCtrlSteps = round(T / controlPeriod);
config_tag = sprintf('T%g_ntheta%d', T, ntheta);

fprintf('Config: %dx%dx%d (%s), T=%.1fs\n', nx, ny, ntheta, config_tag, T);

% -------------------------------------------------------------------------
% 4. Create partitions (as arrays for parfor compatibility)
% -------------------------------------------------------------------------
theta_edges = linspace(obs_lb(3), obs_ub(3), ntheta + 1);
n_reg = ntheta;  % 1x1xN = N regions
reg_lb = zeros(n_reg, 3);
reg_ub = zeros(n_reg, 3);
for it = 1:ntheta
    reg_lb(it, :) = [obs_lb(1), obs_lb(2), theta_edges(it)];
    reg_ub(it, :) = [obs_ub(1), obs_ub(2), theta_edges(it+1)];
end

fprintf('Partitions: %d active\n', n_reg);

% -------------------------------------------------------------------------
% 5. Start parallel pool if available
% -------------------------------------------------------------------------
if ntheta >= 4
    try
        if ~exist('maxWorkers', 'var'), maxWorkers = 24; end
        pool = gcp('nocreate');
        if isempty(pool)
            pool = parpool('local', min(maxWorkers, ntheta));
        elseif pool.NumWorkers ~= min(maxWorkers, ntheta)
            delete(pool);
            pool = parpool('local', min(maxWorkers, ntheta));
        end
        n_workers = pool.NumWorkers;
        fprintf('Parallel pool: %d workers\n', n_workers);
        use_par = true;
    catch
        fprintf('No parallel pool — running sequential.\n');
        use_par = false;
    end
else
    fprintf('ntheta < 4, skipping parfor (overhead > benefit).\n');
    use_par = false;
end

% -------------------------------------------------------------------------
% 6. Main reachability loop
% -------------------------------------------------------------------------
% At each step: NN approx-star reach (with perception uncertainty epsilon)
% → CORA zonotope plant reach → bounding box for next step.
%
% Boxing (get_hypercube_hull / getBox) between steps is necessary because:
%   1. CORA may return multiple Stars from internal splitting — their hull
%      ensures a sound overapproximation for the next step.
%   2. Without boxing, the Star count grows exponentially (~2x every 3-4
%      steps), making propagation intractable beyond ~10 steps.
% This wrapping effect is the fundamental source of conservatism in the
% NNV set-based approach compared to RoVer-CoRe's grid-based HJ method.
fprintf('\nReachability:\n');
t_start = tic;

% Preallocate output storage
% Each step produces n_reg boxes; total = n_reg * (numCtrlSteps + 1)
total_boxes = n_reg * (numCtrlSteps + 1);
all_lb_out = zeros(total_boxes, 3);
all_ub_out = zeros(total_boxes, 3);
all_step_out = zeros(total_boxes, 1);

% Store step 0
all_lb_out(1:n_reg, :) = reg_lb;
all_ub_out(1:n_reg, :) = reg_ub;
all_step_out(1:n_reg) = 0;
write_idx = n_reg + 1;

reachOpt_par.reachMethod = 'approx-star';
if use_par
    reachOpt_par.numCores = 1;                          % parfor handles outer parallelism
else
    if ~exist('maxWorkers', 'var'), maxWorkers = 24; end
    reachOpt_par.numCores = maxWorkers;                  % use all cores for NN reach
end

for k = 1:numCtrlSteps
    step_start = tic;

    new_lb = zeros(n_reg, 3);
    new_ub = zeros(n_reg, 3);

    if use_par
        parfor i = 1:n_reg
            true_lb_i = reg_lb(i, :)';
            true_ub_i = reg_ub(i, :)';

            nn_lb_i = true_lb_i - epsilon;
            nn_ub_i = true_ub_i + epsilon;
            S_nn = Star(nn_lb_i, nn_ub_i);

            [omega_lb_i, omega_ub_i] = nn_reach_bounds_static(Controller, S_nn, reachOpt_par);

            % CORA plant reach from true state
            init_star = Star(true_lb_i, true_ub_i);
            u_star = Star(omega_lb_i, omega_ub_i);

            try
                R = plant.stepReachStar(init_star, u_star);
                if length(R) > 1
                    Rbox = Star.get_hypercube_hull(R);
                else
                    Rbox = R(1).getBox();
                end
                if isempty(Rbox)
                    new_lb(i, :) = true_lb_i';
                    new_ub(i, :) = true_ub_i';
                else
                    new_lb(i, :) = Rbox.lb';
                    new_ub(i, :) = Rbox.ub';
                end
            catch
                new_lb(i, :) = true_lb_i';
                new_ub(i, :) = true_ub_i';
            end
        end
    else
        for i = 1:n_reg
            true_lb_i = reg_lb(i, :)';
            true_ub_i = reg_ub(i, :)';

            nn_lb_i = true_lb_i - epsilon;
            nn_ub_i = true_ub_i + epsilon;
            S_nn = Star(nn_lb_i, nn_ub_i);

            [omega_lb_i, omega_ub_i] = nn_reach_bounds_static(Controller, S_nn, reachOpt_par);

            init_star = Star(true_lb_i, true_ub_i);
            u_star = Star(omega_lb_i, omega_ub_i);

            try
                R = plant.stepReachStar(init_star, u_star);
                if length(R) > 1
                    Rbox = Star.get_hypercube_hull(R);
                else
                    Rbox = R(1).getBox();
                end
                if isempty(Rbox)
                    new_lb(i, :) = true_lb_i';
                    new_ub(i, :) = true_ub_i';
                else
                    new_lb(i, :) = Rbox.lb';
                    new_ub(i, :) = Rbox.ub';
                end
            catch
                new_lb(i, :) = true_lb_i';
                new_ub(i, :) = true_ub_i';
            end
        end
    end

    reg_lb = new_lb;
    reg_ub = new_ub;

    % Store this step
    idx_range = write_idx:(write_idx + n_reg - 1);
    all_lb_out(idx_range, :) = reg_lb;
    all_ub_out(idx_range, :) = reg_ub;
    all_step_out(idx_range) = k;
    write_idx = write_idx + n_reg;

    step_time = toc(step_start);
    fprintf('  Step %d/%d: %d regions (%.1fs), x=[%.2f,%.2f] y=[%.2f,%.2f] th=[%.2f,%.2f]\n', ...
        k, numCtrlSteps, n_reg, step_time, ...
        min(reg_lb(:,1)), max(reg_ub(:,1)), ...
        min(reg_lb(:,2)), max(reg_ub(:,2)), ...
        min(reg_lb(:,3)), max(reg_ub(:,3)));
end

runtime = toc(t_start);
fprintf('\nDone. Runtime: %.2f s\n', runtime);

% -------------------------------------------------------------------------
% 7. Save
% -------------------------------------------------------------------------
results_dir = '../../baselines/results/nnv';
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

save_path = fullfile(results_dir, ['reach_results_' config_tag '.mat']);
save(save_path, ...
    'all_lb_out', 'all_ub_out', 'all_step_out', ...
    'runtime', 'numCtrlSteps', 'T', 'controlPeriod', 'reachStep', ...
    'nx', 'ny', 'ntheta', 'epsilon', 'obs_lb', 'obs_ub', ...
    'config_tag', '-v7.3');

fprintf('Saved: %s\n', save_path);
fprintf('=== Done ===\n');


% =========================================================================
% Helper (must be a regular function for parfor compatibility)
% =========================================================================

function [omega_lb, omega_ub] = nn_reach_bounds_static(Controller, S, reachOpt)
    U_star = Controller.reach(S, reachOpt);
    if isempty(U_star)
        omega_lb = -1; omega_ub = 1; return;
    end
    if length(U_star) > 1
        U_box = Star.get_hypercube_hull(U_star);
    else
        U_box = U_star(1).getBox();
    end
    if isempty(U_box)
        omega_lb = -1; omega_ub = 1;
    else
        omega_lb = max(U_box.lb(1), -1.0);
        omega_ub = min(U_box.ub(1),  1.0);
    end
end
