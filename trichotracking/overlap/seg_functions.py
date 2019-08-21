import numpy as np
from trichotracking.segmentation import removeNoise


def get_segFunctions(distTh, minArea, distIntAreaThresh1=None,
    distIntAreaThresh2=None, distIntAreaThresh3=None,
    distIntAreaThresh4=None, distIntAreaThresh0=None):


    if distIntAreaThresh0 is None:
        def segmentFromDist(img_shape, dist, int):
            """ Segment only based on dist of object and previous overlap."""
            overlap = np.zeros(img_shape).astype(np.uint8)
            overlap[(dist>distTh)] = [255]
            overlap = removeNoise(overlap, minArea)
            return overlap
    else:
        distTh0a, intTh0, distTh0b, areaTh = distIntAreaThresh0
        def segmentFromDistInt(img_shape, dist, int):
            """ Segment based on dist, int and previous overlap."""
            overlap = np.zeros(img_shape).astype(np.uint8)
            overlap[((dist>distTh)
                   |((int>intTh) & (dist>distTh)))] = [255]
            overlap = removeNoise(overlap, areaTh)
            return overlap
    seg_functions = [segmentFromDist]

    if distIntAreaThresh1 is not None:
        intTh1, distTh1, areaTh1 = distIntAreaThresh1
        def segmentFromDistInt(img_shape, dist, int):
            """ Segment based on dist, int and previous overlap."""
            overlap = np.zeros(img_shape).astype(np.uint8)
            overlap[((dist>distTh)
                   |((int>intTh1) & (dist>distTh1)))] = [255]
            overlap = removeNoise(overlap, areaTh1)
            return overlap
        seg_functions.append(segmentFromDistInt)

    if distIntAreaThresh2 is not None:
        intTh2, distTh2, areaTh2 = distIntAreaThresh2
        def segmentFromDistInt2(img_shape, dist, int):
            """ Segment based on dist, int and previous overlap."""
            overlap = np.zeros(img_shape).astype(np.uint8)
            overlap[((dist>distTh)
                   |((int>intTh2) & (dist>distTh2)))] = [255]
            overlap = removeNoise(overlap, areaTh2)
            return overlap
        seg_functions.append(segmentFromDistInt2)

    if distIntAreaThresh3 is not None:
        intTh3a, distTh3a, intTh3b, distTh3b, areaTh3 = distIntAreaThresh3
        def segmentFromDistInt3(img_shape, dist, int):
            """ Segment based on dist, int and previous overlap."""
            overlap = np.zeros(img_shape).astype(np.uint8)
            overlap[((dist>distTh)
                   |((int>intTh3a) & (dist>distTh3a))
                   |((int>intTh3b) & (dist>distTh3b)))] = [255]
            overlap = removeNoise(overlap, areaTh3)
            return overlap
        seg_functions.append(segmentFromDistInt3)

    if distIntAreaThresh4 is not None:
        intTh4a, distTh4a, intTh4b, distTh4b, areaTh4 = distIntAreaThresh4
        def segmentFromDistInt4(img_shape, dist, int):
            """ Segment based on dist, int and previous overlap."""
            overlap = np.zeros(img_shape).astype(np.uint8)
            overlap[((dist>distTh)
                   |((int>intTh4a) & (dist>distTh4a))
                   |((int>intTh4b) & (dist>distTh4b)))] = [255]
            overlap = removeNoise(overlap, areaTh4)
            return overlap
        seg_functions.append(segmentFromDistInt4)


    return seg_functions
