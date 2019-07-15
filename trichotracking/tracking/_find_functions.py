from regionprops import Contour, insideROI

from IPython.core.debugger import set_trace

flags  = ["area", "angle", "bounding_box", "centroid", "contours", "eccentricity", "eigen",
          "form", "length","" "min_box", "min_rect_angle", "orientation", "solidity",
          "pixellist", "aspect_ratio", "min_rect_aspect",
          "min_rect_extent", "perimeter", "dist", "intensities"]


def findTrichomesArea(img, bw, roi=None):
    allObjects = Contour(img, bw, flags=flags, cornerImage=None)
    particles = allObjects.particles
    particles = particles[((particles.area>70)
                          &(particles.area<50000)
                          )]
    if roi is not None:
        inside = insideROI(particles.min_box.values, roi)
        particles = particles[inside]
    return particles


def findTrichomesEccentricty(img, bw, roi=None):
    allObjects = Contour(img, bw, flags=flags, cornerImage=None)
    particles = allObjects.particles
    particles = particles[((particles.area>100)
                          &(particles.area<10000)
                          &(particles.eccentricity>0.9)
                          )]
    if roi is not None:
        inside = insideROI(particles.cx, particles.cy, roi)
        particles = particles[inside]
    return particles
