import os
import sys
#sys.path.append('/Users/Matt/Documents/URI/clam')
#sys.path.append('/Users/Matt/Documents/URI/objrec')
sys.path.append('/home/mdaily/Software/PhD/clam')
sys.path.append('/home/mdaily/Software/PhD/objrec')

import heapq
import numpy as np
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional

from pyclam import Cluster
from pyclam import Manifold
from pyclam import criterion

# Cluster -> (birth-radius, death-radius)
Barcodes = Dict[Cluster, Tuple[float, float]]

class Code:
    def __init__(self, cluster: Cluster, death: float, birth: float = -1):
        self._cluster: Cluster = cluster
        self._death: float = death
        self._birth: float = birth
        return

    def set_birth(self, birth: float):
        self._birth = birth
        return

    @property
    def cluster(self) -> Cluster:
        return self._cluster

    @property
    def death(self) -> float:
        return self._death

    @property
    def birth(self) -> float:
        return self._birth

    @property
    def radius(self) -> float:
        return self._cluster.radius

    def __lt__(self, other: 'Code'):
        return self.cluster.radius > other.cluster.radius


def _normalize(factor: float, barcodes: Barcodes) -> Barcodes:
    return {c: (b / factor, d / factor) for c, (b, d) in barcodes.items()}


def _group_by_cardinality(barcodes: Barcodes) -> Dict[int, Barcodes]:
    cardinalities: List[int] = list(sorted({cluster.cardinality for cluster in barcodes}))
    barcodes_by_cardinality: Dict[int, Barcodes] = {cardinality: dict() for cardinality in cardinalities}
    [barcodes_by_cardinality[cluster.cardinality].update({cluster: lifetime})
     for cluster, lifetime in barcodes.items()]
    return barcodes_by_cardinality


def _merge_high_cardinalities(
        threshold: int,
        barcodes_by_cardinality: Dict[int, Barcodes],
) -> Dict[int, Barcodes]:

    # Merges all barcodes for clusters with cardinality greater than 'threshold'
    high_cardinalities = [v for c, v in barcodes_by_cardinality.items() if c >= threshold]
    if len(high_cardinalities) > 0:
        [high_cardinalities[0].update(h) for h in high_cardinalities[1:]]
        barcodes_by_cardinality = {c: v for c, v in barcodes_by_cardinality.items() if c < threshold}
        barcodes_by_cardinality[threshold] = high_cardinalities[0]
    return barcodes_by_cardinality


def create_barcodes(
        data: np.array,
        *,
        normalize: bool = True,
        merge: Optional[int] = 4,
) -> Dict[int, Barcodes]:
    manifold: Manifold = Manifold(data, 'euclidean').build_tree(criterion.MaxDepth(20))
    barcodes: Barcodes = dict()

    # living-clusters is a heap with highest radius at the top
    living_clusters = [Code(manifold.root, manifold.root.radius)]
    heapq.heapify(living_clusters)

    while living_clusters:  # Go over max-heap
        current: Code = heapq.heappop(living_clusters)

        if current.cluster.children:  # handle children
            current.set_birth(current.radius)
            [left, right] = list(current.cluster.children)

            if left.radius >= current.radius:  # left is still-born
                barcodes[left] = (current.radius, current.radius)
            else:  # or added to living clusters
                heapq.heappush(living_clusters, Code(left, current.radius))

            if right.radius >= current.radius:  # right is still-born
                barcodes[right] = (current.radius, current.radius)
            else:  # or added to living-clusters
                heapq.heappush(living_clusters, Code(right, current.radius))

        else:  # otherwise set birth to zero-radius
            current.set_birth(0.)
        # add current to dict of barcodes
        barcodes[current.cluster] = (current.birth, current.death)

    if normalize:
        barcodes = _normalize(manifold.root.radius, barcodes)

    barcodes_by_cardinality = _group_by_cardinality(barcodes)

    if merge is not None:
        barcodes_by_cardinality = _merge_high_cardinalities(merge, barcodes_by_cardinality)

    return barcodes_by_cardinality

def generateBarcodes(inputPath,
                     outputPath):
    # Read the labels to find out how many shapes there are
    labels = np.loadtxt(os.path.join(inputPath,'Labels.dat'))
    num_shapes = len(labels)
    for index in range(num_shapes):
        inputFile = os.path.join(inputPath,f'Shape{index}.dat')
        shape = np.loadtxt(inputFile,
                           delimiter=',',
                           ndmin=2)
        outputFile = os.path.join(outputPath,f'Shape{index}.bc')
        if (index%100 == 0):
            print('  Create Clam Barcode: ',outputFile)
        with open(outputFile, 'w') as fp:
            barcodes_by_cardinality = create_barcodes(
                shape,
                normalize=True,
                merge=4,
            )
            cardinalities = list(sorted(list(barcodes_by_cardinality.keys())))
            for cardinality in cardinalities:
                barcodes = barcodes_by_cardinality[cardinality]
                barcodes = list(sorted([(birth, death) for birth, death in barcodes.values()]))
                for birth, death in barcodes:
                    fp.write(f'{birth:.4f},{death:.4f},{cardinality}\n')
        fp.close()

        # Now load the barcodes and make the sli from them
        bc = np.loadtxt(outputFile,delimiter=',',ndmin=2)
        
        persistence = bc[:,1] - bc[:,0]
        dims = bc[:,2].astype(int)

        np.savetxt(os.path.join(outputPath,'Labels.dat'),labels);
        
        for dim in np.unique(dims):
            indices = np.nonzero(dims == dim)
            indices = indices[0]

            if (len(indices) == 0):
                print('Error in Clam Barcode For File',inputFile)

            i = np.argsort(persistence[indices])
            indices = indices[i]
            np.savetxt("{}/Shape{}_dim_{}.sli".format(outputPath,index,dim-1),
                       bc[indices,0:2])

