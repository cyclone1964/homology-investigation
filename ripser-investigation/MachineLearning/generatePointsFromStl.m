%generatePointsFromStl - randomly chooses points from an STL defined surface
%
% Points = genratePointsFromStl(FileName, Count) returns a 3xCount matrix, each
% column of which is a point randomly selected from the (presumed 3D)
% surface described in the STL file given. 
function Points = generatePointsFromStl(FileName,Count)

% This function should return a structure with a "faces" and a
% "vertices" member
Surface = stlread(FileName);

% Plot the surface
useNamedFigure('StlSurface'); clf;
patch(Surface);
axis equal;

Faces = Surface.faces';
Vertices = Surface.vertices';

% This only works for triangular tesselations as currently written.
if (size(Faces,1) ~= 3)
    error('Only can do this for triangular tesselations');
end

% Having now done that, we go through all the faces and compute their
% area. To do this we have to rotate each triangle into the x/y
% plane. This is done by forming the cross product of the first two
% sides, converting it to a unit vector, and forming a rotation matrix
% that aligns it with the Z axis.
NumFaces = size(Faces,2);
Areas = zeros(NumFaces,1);

for FaceIndex = 1:NumFaces
    Triangle = Vertices(:,Faces(:,FaceIndex));
    
    Areas(FaceIndex) = 0.5 * norm(cross(Triangle(:,2)-Triangle(:,1), ...
                                        Triangle(:,3)-Triangle(:,1)));
end

% Some of those will have 0 area: we exclude those from the process
Indices = find(Areas > 0);
Areas = Areas(Indices);
Faces = Faces(:,Indices);

% Now we wish to make a vector that allow us to mape a uniform random
% number to discrete selection based upon area. This magic does that
% we think
Selector = cumsum(Areas);
Selector = [0; Selector];
Selector = Selector/Selector(end);

% Now select the faces
FaceIndices = interp1(Selector,1:length(Selector),rand(Count,1),'previous');

% Now, for each face index, find a point randomly placed in that face
Points = zeros(3,Count);
for Index = 1:Count

    Triangle = Vertices(:,Faces(:,FaceIndices(Index)));
    
    % This is a quick and simple way to randomly select a point from the
    % triangle using the "barycentric coordinates". The only question
    % I have here is if the result is truly uniform, but we are doing
    % it this way for now: otherwise I have to iterively select points
    % and then check to see if they are in the triangle, which is a
    % pain in the rear end.
    Points(:,Index) =  Triangle * rand(3,1);
end

