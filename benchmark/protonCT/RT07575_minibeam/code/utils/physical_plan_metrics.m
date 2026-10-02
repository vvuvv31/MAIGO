function metrics = physical_plan_metrics(dose, rois, names, prescription)
%PHYSICAL_PLAN_METRICS Evaluate physical target and OAR doses per fraction.
target = sort(dose(rois{1}), 'descend');
n = numel(target);
metrics = struct('prescription_Gy',prescription, 'target_name',names{1}, ...
    'D95_Gy',target(max(1,round(0.95*n))), ...
    'D98_Gy',target(max(1,round(0.98*n))), ...
    'D50_Gy',target(max(1,round(0.50*n))), ...
    'D2_Gy',target(max(1,round(0.02*n))), 'Dmax_Gy',target(1), ...
    'V100_percent',100*mean(target >= prescription));
metrics.HI = (metrics.D2_Gy-metrics.D98_Gy)/max(metrics.D50_Gy,realmin);
body = unique([rois{1};rois{2}]);
prescriptionVolume = nnz(dose(body) >= prescription);
intersection = nnz(target >= prescription);
metrics.CI_RTOG = prescriptionVolume / n;
metrics.CI_Paddick = intersection^2 / max(n*prescriptionVolume,1);
metrics.rois = struct('name',{},'voxels',{},'mean_Gy',{},'max_Gy',{});
for k = 1:numel(rois)
    if isempty(rois{k})
        continue;
    end
    values = dose(rois{k});
    metrics.rois(end+1) = struct('name',names{k},'voxels',numel(values), ...
        'mean_Gy',mean(values),'max_Gy',max(values)); %#ok<AGROW>
end
end
