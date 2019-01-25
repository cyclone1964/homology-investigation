%plotClusterData - plot the data from the cluster experiments
%
% plotClusterData is a function that will read in the cluster data
% and plot it in three dimensions. We do this in MATLAB because I
% cannot get python to plot in 3 dimensions.
function plotClusterData(FileName,Label)

[Path, Name, Ext] = fileparts(FileName);

if (nargin < 2)
    Label = sprintf('Points In File <%s>',Name);
end

useNamedFigure(Name); clf;
Data = load(FileName);
plot3(Data(:,1),Data(:,2),Data(:,3),'k.');
axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
title(Label);
Filename = [Path,Name,'.png'];    
print('-dpng',FileName);

