function export_physical_plan(outDir, spots, planIndex, weights, dose, grid, options, metrics, solverInfo, historiesPerSpot)
%EXPORT_PHYSICAL_PLAN Export final minibeam weights and physical dose only.
% Weights arrive in DIJ-column order. CSVs retain the original full spot order.
orderedWeights = zeros(height(spots),1);
orderedWeights(planIndex+1) = weights;
spots.weight = orderedWeights;
writetable(spots, fullfile(outDir,'spots_optimized.csv'));
writematrix(orderedWeights, fullfile(outDir,'spot_weights.csv'));
spots.weight = orderedWeights * options.fractions;
writetable(spots, fullfile(outDir,'spots_optimized_total.csv'));
writematrix(spots.weight, fullfile(outDir,'spot_weights_total.csv'));
details = table((0:height(spots)-1)', spots.spot_id, orderedWeights, spots.weight, ...
    'VariableNames', {'plan_index','spot_id','protons_per_fraction','protons_total'});
writetable(details,fullfile(outDir,'spot_weights_details.csv'));
write_dose_mhd(fullfile(outDir,'physical_dose.mhd'),dose,grid);
write_dose_mhd(fullfile(outDir,'physical_dose_total.mhd'),dose*options.fractions,grid);

sumWeights = sum(orderedWeights);
assert(isfinite(sumWeights) && sumWeights > 0, 'Plan weights must have a positive sum.');
manifest = struct('plan_type','proton minibeam', ...
    'histories_per_spot',historiesPerSpot, ...
    'normalization_reference','incident protons before the copper collimator', ...
    'dose_quantity','DoseToMedium','dose_unit','Gy', ...
    'weight_unit','incident protons','primary_particle','proton', ...
    'prescription_per_fraction_Gy',options.prescription_Gy,'fractions',options.fractions, ...
    'spot_count',height(spots),'active_spots',nnz(orderedWeights), ...
    'protons_per_fraction',sumWeights,'protons_total',sumWeights*options.fractions, ...
    'matlab_axis_order','YXZ','mhd_axis_order','XYZ', ...
    'mhd_origin_source','ct.x/y/z first voxel centers; no rotation or reflection', ...
    'optimization_dose_formula','Dij * (weights_in_DIJ_column_order / histories_per_spot)', ...
    'solver',solverInfo,'metrics',metrics);
write_text(fullfile(outDir,'plan_export.json'),jsonencode(manifest,PrettyPrint=true));
end

function write_dose_mhd(path,dose,grid)
% MATLAB arrays are [Y,X,Z]; MetaImage binary storage has X fastest.
[folder,name] = fileparts(path);
rawName = [name '.raw'];
rawPath = fullfile(folder,rawName);
fid = fopen(rawPath,'w','ieee-le');
assert(fid >= 0,'Cannot create %s',rawPath);
cleanup = onCleanup(@() fclose(fid));
count = fwrite(fid,single(permute(dose,[2,1,3])),'single');
assert(count == numel(dose),'Incomplete raw dose export.');
clear cleanup;
spacing = grid.spacing_XYZ_mm;
origin = grid.origin_XYZ_mm;
dim = grid.dimensions_YXZ([2,1,3]);
header = sprintf(['ObjectType = Image\nNDims = 3\nBinaryData = True\n' ...
    'BinaryDataByteOrderMSB = False\nCompressedData = False\n' ...
    'TransformMatrix = 1 0 0 0 1 0 0 0 1\n' ...
    'Offset = %.17g %.17g %.17g\nElementSpacing = %.17g %.17g %.17g\n' ...
    'DimSize = %d %d %d\nElementType = MET_FLOAT\nElementDataFile = %s\n'], ...
    origin,spacing,dim,rawName);
write_text(path,header);
end

function write_text(path,text)
fid = fopen(path,'w');
assert(fid >= 0,'Cannot create %s',path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',text);
end
