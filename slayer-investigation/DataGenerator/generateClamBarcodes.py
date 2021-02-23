import sys
sys.path.append('/home/mdaily/Software/PhD/clam')
sys.path.append('/home/mdaily/Software/PhD/objrec')

from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

from pyclam import Cluster
from pyclam import Manifold

from src.utils import *

# Cluster -> (birth-radius, death-radius)
Barcodes = Dict[Cluster, Tuple[float, float]]

def create_barcodes(
        data: np.array,
        *,
        steps: Optional[int] = 10**3,
        normalize: bool = False,
        merge: Optional[int] = None,
) -> Dict[int, Barcodes]:
    manifold: Manifold = Manifold(data, 'euclidean').build()
    thresholds: np.array = np.linspace(start=manifold.root.radius * (steps - 1) / steps, stop=0, num=steps)
    barcodes: Barcodes = dict()
    living_clusters: Barcodes = {manifold.root: (-1, manifold.root.radius)}
    for threshold in thresholds:
        new_births: Set[Cluster] = set()
        dead_clusters: Set[Cluster] = {cluster for cluster in living_clusters if cluster.radius > threshold}
        while dead_clusters:
            cluster = dead_clusters.pop()
            death = living_clusters.pop(cluster)[1] if cluster in living_clusters else threshold
            barcodes[cluster] = threshold, death
            for child in cluster.children:
                if child.cardinality > 1:
                    (dead_clusters if child.radius > threshold else new_births).add(child)
                else:
                    barcodes[child] = (0, threshold)
        living_clusters.update({cluster: (-1, threshold) for cluster in new_births})

    if normalize:  # normalize radii to [0, 1] range.
        factor = manifold.root.radius
        barcodes = {c: (b / factor, d / factor) for c, (b, d) in barcodes.items()}
    
    cardinalities: List[int] = list(sorted({cluster.cardinality for cluster in barcodes}))
    barcodes_by_cardinality: Dict[int, Barcodes] = {cardinality: dict() for cardinality in cardinalities}
    [barcodes_by_cardinality[cluster.cardinality].update({cluster: lifetime})
     for cluster, lifetime in barcodes.items()]
    
    if merge is not None:
        # Merges all barcodes for clusters with cardinality greater than 'merge'
        high_cardinalities = [v for c, v in barcodes_by_cardinality.items() if c >= merge]
        if len(high_cardinalities) > 0:
            [high_cardinalities[0].update(h) for h in high_cardinalities[1:]]
            barcodes_by_cardinality = {c: v for c, v in barcodes_by_cardinality.items() if c < merge}
            barcodes_by_cardinality[merge] = high_cardinalities[0]
    
    return barcodes_by_cardinality

def generateClamBarcodes(path,
                         num_points: int = 10**2,
                         resolution: int = 10**2,
                         number_per_shape: int = 10**1):
            
    # Read the labels to find out how many shapes there are
    labels = np.loadtxt(os.path.join(path,'Labels.dat'))
    num_shapes = len(labels)
    for index in range(num_shapes):
        filename = os.path.join(path,f'Shape{index}.dat')
        if (index%100 == 0):
            print(filename)
        shape = np.loadtxt(filename,
                           delimiter=',',
                           ndmin=2)
        filename = os.path.join(path,f'Shape{index}.bc')
        with open(filename, 'w') as fp:
            barcodes_by_cardinality = create_barcodes(
                shape,
                steps=resolution,
                normalize=True,
                merge=4,
            )
            cardinalities = list(sorted(list(barcodes_by_cardinality.keys())))
            for cardinality in cardinalities:
                barcodes = barcodes_by_cardinality[cardinality]
                barcodes = list(sorted([(birth, death) for birth, death in barcodes.values()]))
                with open(filename, 'a') as fp:
                    for birth, death in barcodes:
                        fp.write(f'{birth:.4f},{death:.4f},{cardinality}\n')
        fp.close()

        # Now load the barcodes and make the sli from them
        bc = np.loadtxt(filename,delimiter=',',ndmin=2)
        
        persistence = bc[:,1] - bc[:,0]
        dims = bc[:,2].astype(int)
    
        for dim in np.unique(dims):
            indices = np.nonzero(dims == dim)
            indices = indices[0]

            i = np.argsort(persistence[indices])
            indices = indices[i]
            np.savetxt("{}/Shape{}_dim_{}.sli".format(path,index,dim),
                       bc[indices,0:2])

