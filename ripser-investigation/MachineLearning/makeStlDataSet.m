%makeSimpleDataSet - make a simple data set for an initial attempt
%
% makeSimpleDataSet uses the renderSubmarine function to generate
% some data files we can feed to ripser to generate barcodes for
% highlights on a submarine
function makeStlDataSet

% We now randomly choose 512 points from that
Points = generatePointsFromStl('Submarine.stl',512);

% Now, we form the lower distance matrix of the distances
FID = fopen('SubmarinePoints.ldm','w');

for First = 2:size(Points,2)
    for Second = 1:(First-1)
        fprintf(FID,'%.6e,',norm(Points(:,First) - ...
                                 Points(:,Second)));
    end
    fprintf(FID,'\n');
end
fclose(FID);

% Now, let's instead randomly select vertices
Sub = stlread('Submarine.stl');
Points = Sub.vertices(randi(length(Sub.vertices),512));

% Now, we form the lower distance matrix of the distances
FID = fopen('SubmarineVertices.ldm','w');

for First = 2:size(Points,2)
    for Second = 1:(First-1)
        fprintf(FID,'%.6e,',norm(Points(:,First) - ...
                                 Points(:,Second)));
    end
    fprintf(FID,'\n');
end
fclose(FID);
