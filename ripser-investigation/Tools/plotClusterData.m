%plotClusterData - plot the data from the cluster experiments
%
% plotClusterData is a function that will read in the cluster data
% and plot it in three dimensions. We do this in MATLAB because I
% cannot get python to plot in 3 dimensions.
function plotClusterData

ClusterData = {'Balls-1','Cluster Cube (Separation = 1)'
               'Balls-2','Cluster Cube (Separation = 2)'
               'Balls-4','Cluster Cube (Separation = 4)'
               'Balls-6','Cluster Cube (Separation = 6)'
               'Balls-8','Cluster Cube (Separation = 8)'
               'Balls-16','Cluster Cube (Separation = 16)'
               'Balls-64','Cluster Cube (Separation = 64)'
               'Line','Tube Of Points (Radius = 1)'
               'Helix-5','Helix of Points'};

for Index =1:size(ClusterData,1)
    figure(Index); clf;
    Data = load(['../Output/' ClusterData{Index,1} '.dat']);
    plot3(Data(:,1),Data(:,2),Data(:,3),'k.');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(ClusterData{Index,2});
    
    FileName = ['../Output/' ClusterData{Index,1} '.png'];
    print('-dpng',FileName);
end