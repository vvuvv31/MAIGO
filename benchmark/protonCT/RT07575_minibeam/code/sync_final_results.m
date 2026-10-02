function sync_final_results(outDir, syncCodeDir)
%SYNC_FINAL_RESULTS Copy only completed dose/weight exports to the minimal bundle.
% May be called again without rerunning ADMM if a previous copy failed.
codeDir = fileparts(mfilename('fullpath'));
if nargin < 1
    outDir = fullfile(codeDir, 'output_minibeam');
end
if nargin < 2
    syncCodeDir = '/mnt/sdb/wuwei/MAIGO_minibeam/benchmark/protonCT/RT07575_minibeam/code';
end
files = {'minibeam_dose_result.mat', 'physical_dose.mhd', 'physical_dose.raw', ...
    'physical_dose_total.mhd', 'physical_dose_total.raw', ...
    'spots_optimized.csv', 'spots_optimized_total.csv', ...
    'spot_weights.csv', 'spot_weights_total.csv', 'spot_weights_details.csv', ...
    'plan_export.json'};
for k = 1:numel(files)
    assert(isfile(fullfile(outDir, files{k})), 'Missing final output: %s', files{k});
end
destination = fullfile(syncCodeDir, 'output_minibeam');
if ~isfolder(destination)
    [ok, message] = mkdir(destination);
    assert(ok, 'Cannot create sync directory: %s', message);
end
[okSource, sourceInfo] = fileattrib(outDir);
[okDestination, destinationInfo] = fileattrib(destination);
assert(okSource && okDestination, 'Cannot resolve output directories.');
if strcmp(sourceInfo.Name, destinationInfo.Name)
    fprintf('Final outputs are already in the synchronized bundle: %s\n', destination);
    return;
end
for k = 1:numel(files)
    [ok, message] = copyfile(fullfile(outDir, files{k}), fullfile(destination, files{k}), 'f');
    assert(ok, 'Sync failed; local results are retained in %s: %s', outDir, message);
end
fprintf('Final minibeam outputs synchronized: %s\n', destination);
end
