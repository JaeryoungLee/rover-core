% verify_individual_propagation.m
%
% NNV baseline with individual Star propagation (no boxing between steps).
% When CORA splits the reachable set into multiple Stars, each is
% propagated independently at the next step.
%
% Usage:
%   T_arg=1; ntheta_arg=1; verify_individual_propagation

fprintf('=== NNV: Individual Star Propagation ===\n');

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
plant = NonLinearODE(3, 1, dynamics, reachStep, controlPeriod, eye(3));
plant.stepReachStar(Star([15.5; -0.5; 0.0], [16.0; 0.0; 0.1]), Star(0, 0));
fprintf('  CORA ready.\n');

% -------------------------------------------------------------------------
% 3. Configuration
% -------------------------------------------------------------------------
obs_lb = [15; -1; -pi];
obs_ub = [17;  1;  pi];
epsilon = [0.5; 0.5; 0.1];

if ~exist('ntheta_arg', 'var'), ntheta_arg = 1; end
if ~exist('T_arg', 'var'), T_arg = 1.0; end
ntheta = ntheta_arg;
T = T_arg;
numCtrlSteps = round(T / controlPeriod);
config_tag = sprintf('indiv_T%g_ntheta%d', T, ntheta);

fprintf('Config: ntheta=%d, T=%.1fs (%d steps)\n', ntheta, T, numCtrlSteps);

% -------------------------------------------------------------------------
% 4. Create initial partitions
% -------------------------------------------------------------------------
theta_edges = linspace(obs_lb(3), obs_ub(3), ntheta + 1);
reg_stars = cell(ntheta, 1);
for it = 1:ntheta
    lb = [obs_lb(1); obs_lb(2); theta_edges(it)];
    ub = [obs_ub(1); obs_ub(2); theta_edges(it+1)];
    reg_stars{it} = Star(lb, ub);
end

fprintf('Initial Stars: %d\n', length(reg_stars));

reachOpt.reachMethod = 'approx-star';
reachOpt.numCores = 1;

% -------------------------------------------------------------------------
% 5. Storage (dynamic — grows as Stars split)
% -------------------------------------------------------------------------
all_lb_list = {};
all_ub_list = {};
all_step_list = {};

% Store step 0
for i = 1:length(reg_stars)
    b = reg_stars{i}.getBox();
    all_lb_list{end+1} = b.lb';
    all_ub_list{end+1} = b.ub';
    all_step_list{end+1} = 0;
end

% -------------------------------------------------------------------------
% 6. Main reachability loop
% -------------------------------------------------------------------------
% Start parallel pool
if ~exist('maxWorkers', 'var'), maxWorkers = 24; end
try
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local', maxWorkers);
    end
    fprintf('Parallel pool: %d workers\n', pool.NumWorkers);
catch
    fprintf('No parallel pool — running sequential.\n');
end

fprintf('\nReachability:\n');
t_start = tic;

for k = 1:numCtrlSteps
    step_start = tic;
    n_in = length(reg_stars);

    % parfor: each Star produces a cell of output Stars and boxes
    out_stars = cell(n_in, 1);
    out_lb = cell(n_in, 1);
    out_ub = cell(n_in, 1);

    parfor i = 1:n_in
        current_star = reg_stars{i};

        % Get bounding box for NN input
        current_box = current_star.getBox();
        true_lb_i = current_box.lb;
        true_ub_i = current_box.ub;

        % NN input with perception uncertainty
        nn_lb_i = true_lb_i - epsilon;
        nn_ub_i = true_ub_i + epsilon;
        S_nn = Star(nn_lb_i, nn_ub_i);

        [omega_lb_i, omega_ub_i] = nn_reach_bounds_static(Controller, S_nn, reachOpt);

        % Plant reach from current Star
        u_star = Star(omega_lb_i, omega_ub_i);

        try
            R = plant.stepReachStar(current_star, u_star);
            stars_i = cell(length(R), 1);
            lb_i = zeros(length(R), 3);
            ub_i = zeros(length(R), 3);
            for j = 1:length(R)
                stars_i{j} = R(j);
                Rbox = R(j).getBox();
                lb_i(j, :) = Rbox.lb';
                ub_i(j, :) = Rbox.ub';
            end
            out_stars{i} = stars_i;
            out_lb{i} = lb_i;
            out_ub{i} = ub_i;
        catch
            out_stars{i} = {current_star};
            out_lb{i} = true_lb_i';
            out_ub{i} = true_ub_i';
        end
    end

    % Flatten results
    new_stars = {};
    for i = 1:n_in
        for j = 1:length(out_stars{i})
            new_stars{end+1} = out_stars{i}{j};
        end
        lb_i = out_lb{i};
        for j = 1:size(lb_i, 1)
            all_lb_list{end+1} = lb_i(j, :);
            all_ub_list{end+1} = out_ub{i}(j, :);
            all_step_list{end+1} = k;
        end
    end

    reg_stars = new_stars';
    n_out = length(reg_stars);

    step_time = toc(step_start);
    elapsed = toc(t_start);
    avg_per_step = elapsed / k;
    eta = avg_per_step * (numCtrlSteps - k);
    fprintf('  Step %d/%d: %d->%d Stars (%.1fs) [elapsed=%.0fs, ETA=%.0fs]\n', ...
        k, numCtrlSteps, n_in, n_out, step_time, elapsed, eta);
end

runtime = toc(t_start);
fprintf('\nDone. Runtime: %.2f s, final Stars: %d\n', runtime, length(reg_stars));

% -------------------------------------------------------------------------
% 7. Save (same format as verify_reversed_dubins.m)
% -------------------------------------------------------------------------
n_total = length(all_lb_list);
all_lb_out = zeros(n_total, 3);
all_ub_out = zeros(n_total, 3);
all_step_out = zeros(n_total, 1);
for i = 1:n_total
    all_lb_out(i, :) = all_lb_list{i};
    all_ub_out(i, :) = all_ub_list{i};
    all_step_out(i) = all_step_list{i};
end

results_dir = '../../baselines/results/nnv';
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

nx = 1; ny = 1;
save_path = fullfile(results_dir, ['reach_results_' config_tag '.mat']);
save(save_path, ...
    'all_lb_out', 'all_ub_out', 'all_step_out', ...
    'runtime', 'numCtrlSteps', 'T', 'controlPeriod', 'reachStep', ...
    'nx', 'ny', 'ntheta', 'epsilon', 'obs_lb', 'obs_ub', ...
    'config_tag', '-v7.3');

fprintf('Saved: %s\n', save_path);
fprintf('=== Done ===\n');


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
