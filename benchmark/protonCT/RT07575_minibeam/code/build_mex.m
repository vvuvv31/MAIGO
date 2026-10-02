function build_mex()
%BUILD_MEX Compile the supplied ADMM/CG C sources for the current platform.
utilsDir = fullfile(fileparts(mfilename('fullpath')), 'utils');
names = {'CG033114', 'JtJ_id'};
for k = 1:numel(names)
    clear(names{k});
    % These sources use separate real/imaginary single-precision arrays.
    mex('-R2017b', '-O', '-outdir', utilsDir, ...
        fullfile(utilsDir, [names{k} '.c']));
end
rehash;
fprintf('Compiled ADMM solver: %s\n', mexext);
end
