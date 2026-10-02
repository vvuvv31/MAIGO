function resultFile = run_admm_minibeam(options)
%RUN_ADMM_MINIBEAM Optimize the completed proton minibeam DIJ (10 ADMM iterations).
% Optimization variable x is the number of DIJ history batches per spot.
% Exported weights are incident proton counts, not MU or batch multipliers.
if nargin == 0
    options = struct();
end
codeDir = fileparts(mfilename('fullpath'));
caseDir = fileparts(codeDir);
addpath(fullfile(codeDir, 'utils'));
defaults = struct('prescription_Gy', 4, 'fractions', 16, ...
    'N_iter', 10, 'cg_iter', 20, 'cg_tol', 1e-6, ...
    'normalize_D95', true, ...
    'output_dir', fullfile(codeDir, 'output_minibeam'), ...
    'ct_file', fullfile(codeDir, 'RT07575_cropXY_bodyZgtv20mm.mat'), ...
    'dij_file', fullfile(caseDir, 'merged', 'dij.mat'), ...
    'sync_code_dir', '/mnt/sdb/wuwei/MAIGO_minibeam/benchmark/protonCT/RT07575_minibeam/code');
fields = fieldnames(defaults);
for k = 1:numel(fields)
    if ~isfield(options, fields{k})
        options.(fields{k}) = defaults.(fields{k});
    end
end
validateattributes(options.prescription_Gy, {'numeric'}, {'scalar','finite','positive'});
validateattributes(options.fractions, {'numeric'}, {'scalar','integer','positive'});
validateattributes(options.N_iter, {'numeric'}, {'scalar','integer','positive'});
validateattributes(options.cg_iter, {'numeric'}, {'scalar','integer','positive'});
validateattributes(options.cg_tol, {'numeric'}, {'scalar','finite','>',0,'<',1});
validateattributes(options.normalize_D95, {'logical'}, {'scalar'});
if exist('CG033114', 'file') ~= 3 || exist('JtJ_id', 'file') ~= 3
    build_mex();
end
if ~isfolder(options.output_dir)
    mkdir(options.output_dir);
end

fprintf('Loading CT: %s\n', options.ct_file);
patient = load(options.ct_file, 'ct', 'cst');
ct = patient.ct;
cst = patient.cst;
clear patient;
fprintf('Loading physical sparse DIJ: %s\n', options.dij_file);
input = load(options.dij_file, 'Dij', 'cube_dimensions', ...
    'histories_per_spot', 'plan_index');
assert(issparse(input.Dij), 'Physical DIJ must remain sparse.');
dimensions = double(ct.cubeDim(:)');
assert(isequal(double(input.cube_dimensions(:)'), dimensions), ...
    'DIJ cube dimensions do not match the CT.');
[nY, nX] = size(input.Dij);
assert(nY == prod(dimensions), 'DIJ voxel count does not match the CT.');
historiesPerSpot = double(input.histories_per_spot);
validateattributes(historiesPerSpot, {'numeric'}, {'scalar','finite','integer','positive'});
assert(historiesPerSpot == 1000000, ...
    'Expected the minibeam DIJ: 1,000,000 incident protons per spot.');
planIndex = double(input.plan_index(:));
assert(numel(planIndex) == nX && all(isfinite(planIndex)) && ...
    all(planIndex >= 0 & planIndex == fix(planIndex)) && ...
    numel(unique(planIndex)) == nX, 'Invalid DIJ plan_index.');
metadata = jsondecode(fileread(fullfile(fileparts(options.dij_file), 'dij.json')));
assert(strcmp(metadata.voxel_order_name, 'matrad'), 'DIJ must use matRad Y-fastest rows.');
assert(metadata.histories_per_spot == historiesPerSpot, 'DIJ history metadata mismatch.');
assert(strcmp(metadata.status, 'accepted'), 'DIJ assembly is not accepted.');
complete = jsondecode(fileread(fullfile(fileparts(options.dij_file), 'COMPLETE.json')));
assert(strcmp(complete.status, 'accepted') && complete.spots == nX && ...
    isequal(double(complete.matrix_shape(:)'), [nY,nX]) && ...
    complete.histories == nX*historiesPerSpot && complete.nonzeros == nnz(input.Dij), ...
    'Expected the complete accepted minibeam DIJ.');
assert(strcmp(metadata.dose_unit, 'Gy') && nX == 870 && ...
    isequal([metadata.voxel_size_x_mm,metadata.voxel_size_y_mm,metadata.voxel_size_z_mm], ...
    [ct.resolution.x,ct.resolution.y,ct.resolution.z]), ...
    'Minibeam DIJ units, spot count or voxel spacing mismatch.');
assert(numel(metadata.columns) == nX && ...
    isequal(double([metadata.columns.plan_index]'), planIndex), ...
    'DIJ MAT and JSON column mappings disagree.');
Dij = {input.Dij};
clear input;

spots = readtable(fullfile(caseDir, 'spots.csv'), 'VariableNamingRule', 'preserve');
assert(nX == height(spots) && isequal(sort(planIndex), (0:nX-1)'), ...
    'This export requires the complete spot plan.');
columnSpots = spots(planIndex + 1, :);
assert(isequal(double(columnSpots.spot_id), str2double(string({metadata.columns.spot_id}'))), ...
    'DIJ column spot IDs disagree with spots.csv.');
assert(max(abs(columnSpots.energy_MeV - double([metadata.columns.energy_mev]'))) < 1e-8 && ...
    max(abs(columnSpots.x_mm - double([metadata.columns.x_mm]'))) < 1e-8 && ...
    max(abs(columnSpots.y_mm - double([metadata.columns.y_mm]'))) < 1e-8, ...
    'Spot positions or energies disagree with DIJ metadata.');

names = {'GTV', 'BODY', 'Brain Stem', 'Spinal cord', 'Eye L', 'Eye R', 'Lens L', 'Lens R'};
c = cell(numel(names), 1);
for k = 1:numel(names)
    row = find(strcmpi(cst(:,2), names{k}));
    assert(isscalar(row), 'Missing or ambiguous structure: %s', names{k});
    c{k} = unique(double(cst{row,4}{1}(:)));
    assert(all(c{k} >= 1 & c{k} <= nY & c{k} == fix(c{k})), ...
        'Structure indices are invalid: %s', names{k});
    if isempty(c{k})
        fprintf('Skipping empty structure: %s\n', names{k});
    end
end
assert(~isempty(c{1}) && ~isempty(c{2}), 'Target and BODY must not be empty.');
assert(all(ismember(c{1}, c{2})), 'Target must lie inside BODY.');
c{2} = setdiff(c{2}, c{1});
n_c = cellfun(@numel, c);
px = options.prescription_Gy;
nfrac = options.fractions;
% Keep the previous prescription numbers and objective weights, now in Gy.
w_obj = [1;1;1;0.1;0.1;1;0.2;1;0.2;1;0.2;1;0.2;1;0.2;1;0.2];
s_obj = [px;px;px;0;px;54/nfrac;0;45/nfrac;0;50/nfrac;0;50/nfrac;0;10/nfrac;0;10/nfrac;0];
c_obj = [1;1;1;2;2;3;3;4;4;5;5;6;6;7;7;8;8];
type_obj = [0;2;3;0;3;3;0;3;0;3;0;3;0;3;0;3;0];
n_obj = nan(17,1);
n_obj(2) = max(1, round(n_c(1)*0.95));
keep = n_c(c_obj) > 0;
w_obj = w_obj(keep); s_obj = s_obj(keep);
c_obj = c_obj(keep); type_obj = type_obj(keep); n_obj = n_obj(keep);
id_obj = c(c_obj);
para = struct('Dij', {Dij}, 'nX', nX, 'nY', nY, 'N_c', numel(c), ...
    'n_c', n_c, 'c', {c}, 'N_obj', numel(w_obj), 'N_dij', 1, ...
    'w_obj', w_obj, 's_obj', s_obj, 'c_obj', c_obj, ...
    'type_obj', type_obj, 'n_obj', n_obj, 'id_obj', {id_obj});
AmX = @AmX_v3; AtmX = @AtmX_v3;
x0 = ones(nX,1,'single');
maxAtA = double(norm(AtmX(AmX(x0,para),para)) / norm(x0));
assert(isfinite(maxAtA) && maxAtA > 0, 'DIJ/objective normal operator has no positive scale.');
ip = struct('N_iter', options.N_iter, 'mu_min', 0, 'mup', single(0.01*maxAtA), ...
    'relative_tol', 1e-3, 'absolute_tol', 1e-5, 'objective_tol', 1e-5, ...
    'minimum_iter', 10);
var_CG = struct('id_x12', uint32(0), 'id_xs', uint32(1), 'id_xr', uint32(0), ...
    'JtJ', 'JtJ_id', 'cg_iter', options.cg_iter, 'cg_tol', options.cg_tol, ...
    'AmX', AmX, 'AtmX', AtmX, 'var_AtA', para, 'mu_xs', ip.mup);
fprintf('Optimizing minibeam: %d spots, target %s, %.4g Gy/fraction, %d fractions.\n', nX, names{1}, px, nfrac);
timer = tic;
[x, solverInfo, history] = admm_mmu1(ip, var_CG);
solverInfo.elapsed_s = toc(timer);

d = Dij{1} * double(x);
beforeNormalization = physical_plan_metrics(d, c, names, px);
normalization = 1;
if options.normalize_D95
    assert(beforeNormalization.D95_Gy > 0, 'Cannot normalize a zero target dose.');
    normalization = px / beforeNormalization.D95_Gy;
end
batchWeights = double(x) * normalization;
% Compute from the exact exported weights after final normalization.
d = Dij{1} * batchWeights;
weights = batchWeights * historiesPerSpot;
metrics = physical_plan_metrics(d, c, names, px);
metrics.normalization_factor = normalization;
metrics.before_normalization = beforeNormalization;
physicalDose = reshape(d, dimensions);
physicalDoseTotal = physicalDose * nfrac;
weightsTotal = weights * nfrac;
grid = struct('dimensions_YXZ', dimensions, ...
    'spacing_XYZ_mm', [ct.resolution.x,ct.resolution.y,ct.resolution.z], ...
    'origin_XYZ_mm', [ct.x(1),ct.y(1),ct.z(1)], 'x_mm',ct.x,'y_mm',ct.y,'z_mm',ct.z);
resultGUI = struct('physicalDose', physicalDose, 'w', weights, ...
    'description', 'Minibeam ADMM physical DoseToMedium; w is incident protons per fraction');
resultFile = fullfile(options.output_dir, 'minibeam_dose_result.mat');
save(resultFile, 'resultGUI', 'physicalDose', 'physicalDoseTotal', ...
    'weights', 'weightsTotal', 'batchWeights', 'planIndex', 'historiesPerSpot', ...
    'grid', 'metrics', 'solverInfo', 'history', 'options', 'metadata', '-v7.3');
export_physical_plan(options.output_dir, spots, planIndex, weights, ...
    physicalDose, grid, options, metrics, solverInfo, historiesPerSpot);
fprintf('Physical dose exported: %s\n', resultFile);
fprintf('Target D95 %.6g Gy, D2 %.6g Gy; %d active spots; solver converged=%d.\n', ...
    metrics.D95_Gy, metrics.D2_Gy, nnz(weights), solverInfo.converged);
if ~isempty(options.sync_code_dir)
    sync_final_results(options.output_dir, options.sync_code_dir);
end
end
