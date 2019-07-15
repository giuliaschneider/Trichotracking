import numpy as np
import numpy.linalg
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from .contour_functions import (calcArea,
                                calcBoundingBox,
                                calcConvexArea,
                                calcCentroid,
                                calcEigenvalues,
                                calcEllipse,
                                calcLength,
                                calcMinRect,
                                calcPerimeter,
                                calcPixellist,
                                calcSolidity,
                                getAngleFromMoments,
                                getExtremes)


from IPython.core.debugger import set_trace


# Index of x, y coordinates in numpy array
NP_YCOORD = 0
NP_XCOORD = 1


class Contour:
    """ Extracts the contours in the black-white image bw.
        The method calcProperties calculates the flagged properties and
        saves them in a dataframe self.particles

        Properties:

        area            # area of the region
        perimeter       # perimeter of the region
        form
        moments         # values of moments as  dict
        centroid        # the centroid of the region as cx, cy
        weighted_centroid the centroid of the weighted region as cx, cy
        pixelList       # array of indices of on-pixels of contour
        bounding_box    # the bounding box parameters (x,y,width,height)
        bx
        by
        bw
        bh
        aspect_ratio    # ratio of width to height
        min_rect_ascpect
        equi_diameter   # d of circle with same as area
        extent          # contour area/bounding box area
        convex_hull     # convex hull of the region
        convex_area     # area of the convex hull
        solidity        # contour area / convex hull area
        majoraxis_length    # length of major axis
        minoraxis_length    # length of minor axis
        orientation     # orientation of ellipse
        eccentricity    # eccentricity of ellipse
        leftmost        # leftmost point of the contour
        rightmost       # rightmost point of the contour
        topmost         # topmost point of the contour
        bottommost      # bottommost point of the contour
        """

    def __init__(self, img, bw_img, flags=None, cornerImage=None,
                 weigthing_function=None):
        """ Finds the contours, saves properties in dataframe particles."""
        # Initialize variables
        self.img = img
        self.bw_img = bw_img
        self.flags = flags
        self.cornerImage = cornerImage
        self.weigthing_function = weigthing_function
        if self.flags is None:
            self.flags = ["area", "perimeter", "form", "equivalent_diameter",
                          "moments", "centroid", "min_rect", "length",
                          "bounding_box", "aspect_ratio", "extent",
                          "min_rect_extent",
                          "convex_hull", "solidity", "orientation",
                          "length_majoraxis", "length_minoraxis",
                          "eccentricity", "extremes", "pixellist",
                          "corners"]
        if self.weigthing_function is None:
            self.weigthing_function = lambda I: I
        self.particles = pd.DataFrame()

        # Find contours
        self.getContours()
        self.nContours = len(self.contours)
        if("contours" in self.flags):
            self.particles["contours"] = self.contours

        # Calculate properties
        if self.nContours==0:
            self.calcRegionpropsEmpty()
        else:
            self.calcRegionprops()

    def getContours(self):
        """ Extracts contours from bw."""
        img, contours, hierarchy = cv2.findContours(self.bw_img,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours

    def getFilledImage(self):
        self.filledImage = np.zeros(self.img.shape[0:2])
        for i, c in enumerate(self.contours):
            cv2.drawContours(self.filledImage, [c], 0, (i+1), -1)
        # Draw contours fill holes,
        nlabels, mask = cv2.connectedComponents(self.bw_img, 8, cv2.CV_32S)
        mask[mask>0] = [255]
        mask = mask.astype(np.uint8)
        self.filledImage = cv2.bitwise_and(self.filledImage,
                                           self.filledImage, mask = mask)

    def getCornerImage(self):
        if self.cornerImage is None:
            harris_corners = cv2.cornerHarris(self.img, 2, 3, 0.04)
            self.cornerImage = np.zeros(self.img.shape[0:2],np.uint8)
            self.cornerImage[harris_corners>0.02*harris_corners.max()]=[255]

    def getDistImage(self):
        h, w = self.bw_img.shape[:2]
        bwpadded = np.zeros((h+2, w+2)).astype(np.uint8)
        bwpadded[1:-1,1:-1] = self.bw_img
        dist = cv2.distanceTransform(bwpadded, cv2.DIST_L2, 3)
        self.distImage = dist[1:-1,1:-1]

    def calcRegionprops(self):
        """ Calculates the flagged properties of the contours."""

        # Area
        if ("area" in self.flags):
            area = calcArea(self.contours)
            self.particles["area"] = area
        else:
            area = None

        # Perimeter
        if ("perimeter" in self.flags):
            perimeter = calcPerimeter(self.contours)
            self.particles["perimeter"] = perimeter
        else:
            perimeter = None

        # Form
        if ("form" in self.flags):
            if area is None:
                area = calcArea(self.contours)
            if perimeter is None:
                perimeter = calcPerimeter(self.contours)
            a = 4*np.pi*area
            b = perimeter**2
            form = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            self.particles["form"] = form

        # Equivalent diameter
        if ("equivalent_diameter" in self.flags):
            if area is None:
                area = calcArea(self.contours)
            equi_diameter= np.sqrt(4*area/np.pi)
            self.particles["equivalent_diameter"] = equi_diameter

        # Moments
        if ("moments" in self.flags):
            moments = [cv2.moments(c) for c in self.contours]
            self.particles["moments"] = moments
        else:
            moments = None

        # Centroid
        if ("centroid" in self.flags):
            if moments is None:
                moments = [cv2.moments(c) for c in self.contours]
            cx, cy = calcCentroid(moments)
            self.particles["cx"] = cx
            self.particles["cy"] = cy

        # Min Rectangle
        if ("min_rect" in self.flags):
            min_rect, min_box, min_rect_angle = calcMinRect(self.contours)
            self.particles["min_rect"] = min_rect
        else:
            min_rect = None
            min_box = None
            min_rect_angle = None

        if ("min_box") in self.flags:
            if min_box is None:
                min_rect, min_box, min_rect_angle = calcMinRect(self.contours)
            self.particles["min_box"] = min_box

        # Min Rectangle
        if("min_rect_angle" in self.flags):
            if min_rect_angle is None:
                min_rect, min_box, min_rect_angle = calcMinRect(self.contours)
            self.particles["min_rect_angle"] = min_rect_angle

        # Aspect ratio
        if ("min_rect_aspect" in self.flags):
            if min_box is None:
                min_rect, min_box, min_rect_angle = calcMinRect(self.contours)
            min_box = np.array(min_box)
            lr = numpy.linalg.norm(min_box[:,0,:]-min_box[:,1,:],axis=1)
            tb = numpy.linalg.norm(min_box[:,0,:]-min_box[:,2,:],axis=1)
            bfw = np.max(np.vstack((lr, tb)),axis=0)
            bfh = np.min(np.vstack((lr, tb)),axis=0)
            min_rect_aspect = np.divide(bfw, bfh, out=np.zeros_like(bfw),
                                        where=bfh!=0)
            self.particles["min_rect_aspect"] = min_rect_aspect

        # Min rect Extent
        if ("min_rect_extent" in self.flags):
            if area is None:
                area = calcArea(self.contours)
            if min_box is None:
                min_rect, min_box, min_rect_angle = calcMinRect(self.contours)
            min_box = np.array(min_box)
            lr = numpy.linalg.norm(min_box[:,0,:]-min_box[:,1,:],axis=1)
            tb = numpy.linalg.norm(min_box[:,0,:]-min_box[:,2,:],axis=1)
            bfw = np.max(np.vstack((lr, tb)),axis=0)
            bfh = np.min(np.vstack((lr, tb)),axis=0)
            b = bfw*bfh
            min_rect_extent = np.divide(area, b, out=np.zeros_like(area),
                                where=b!=0)
            self.particles["min_rect_extent"] = min_rect_extent

        # Length
        if ("length" in self.flags):
            if min_box is None:
                min_rect, min_box, min_rect_angle = calcMinRect(self.contours)
            length = calcLength(min_box)
            self.particles["length"] = length

        # Bounding box
        if ("bounding_box" in self.flags):
            bounding_box, bx, by, bw, bh = calcBoundingBox(self.contours)
            self.particles["bx"] = bx
            self.particles["by"] = by
            self.particles["bw"] = bw
            self.particles["bh"] = bh
        else:
            bounding_box = None

        # Aspect ratio
        if ("aspect_ratio" in self.flags):
            if bounding_box is None:
                bounding_box, bx, by, bw, bh = calcBoundingBox(self.contours)
            bfw, bfh = bw.astype('float'), bh.astype('float')
            aspect_ratio = np.divide(bfw, bfh,out=np.zeros_like(bfw), where=bfh!=0)
            self.particles["aspect_ratio"] = aspect_ratio

        # Extent
        if ("extent" in self.flags):
            if area is None:
                area = calcArea(self.contours)
            if bounding_box is None:
                bounding_box, bx, by, bw, bh = calcBoundingBox(self.contours)
            b = bw*bh
            extent = np.divide(area, b, out=np.zeros_like(area),
                                where=b!=0)
            self.particles["extent"] = extent


        # Convex Hull
        if ("convex_hull" in self.flags):
            convex_hull = [cv2.convexHull(c) for c in self.contours]
            self.particles["convex_hull"] = convex_hull
        else:
            convex_hull = None

        # Convex area
        if ("convex_area" in self.flags):
            if convex_hull is None:
                convex_hull = [cv2.convexHull(c) for c in self.contours]
            convex_area = calcConvexArea(convex_hull)
            self.particles["convex_area"] = convex_area
        else:
            convex_area = None

        # Solidity
        if ("solidity" in self.flags):
            solidity, area, convex_hull, convex_area = \
                calcSolidity(self.contours, area, convex_hull, convex_area)
            self.particles["solidity"] = solidity
        else:
            solidity = None

        # Ellipse
        if (("orientation" in self.flags) or ("eccentricity" in self.flags)
            or ("majoraxis_length" in self.flags)
            or ("minoraxis_length" in self.flags)):
            orientation, majoraxis_length, minoraxis_length, eccentricity \
             = calcEllipse(self.contours)

            # Orientation
            if ("orientation" in self.flags):
                self.particles["orientation"] = orientation

            # majoraxis_length
            if ("majoraxis_length" in self.flags):
                self.particles["majoraxis_length"] = majoraxis_length

            # minoraxis_length
            if ("minoraxis_length" in self.flags):
                self.particles["minoraxis_length"] = minoraxis_length

            # eccentricity
            if ("eccentricity" in self.flags):
                self.particles["eccentricity"] = eccentricity
        else:
            orientation, eccentricity = None, None
            majoraxis_length, minoraxis_length = None, None

        # Angle
        if("angle" in self.flags):
            theta = getAngleFromMoments(self.contours)
            self.particles['angle'] = theta


        # Extremes
        if("extremes" in self.flags):
            leftmost, rightmost, topmost, bottommost = getExtremes(self.contours)
            self.particles["leftmost"] = leftmost
            self.particles["rigthmost"] = rightmost
            self.particles["topmost"] = topmost
            self.particles["bottommost"] = bottommost


        # Pixellist
        if("pixellist" in self.flags):
            if bounding_box is None:
                bounding_box, bx, by, bw, bh = calcBoundingBox(self.contours)
            self.getFilledImage()
            pixellist_ycoord, pixellist_xcoord = calcPixellist(
                        self.filledImage, self.contours, bx, by, bw, bh)
            self.particles["pixellist_xcoord"] = pixellist_xcoord
            self.particles["pixellist_ycoord"] = pixellist_ycoord
        else:
            pixellist_xcoord = None
            pixellist_ycoord = None
            self.filledImage = None



        # Corners
        if("corners" in self.flags):
            if bounding_box is None:
                bounding_box, bx, by, bw, bh = calcBoundingBox(self.contours)
            cornerList, cornerPts = calcCorners(self.filledImage,
                                        self.cornerImage, bx, by, bw, bh)
            self.particles["corners"] = cornerList
        else:
            cornerList = None

        # Weighed centroid
        if("weighted_centroid" in self.flags):
            if bounding_box is None:
                bounding_box, bx, by, bw, bh = calcBoundingBox(self.contours)
            if pixellist_xcoord is None or pixellist_xcoord is None:
                if self.filledImage is None:
                    self.getFilledImage()
                pixellist_ycoord, pixellist_xcoord = calcPixellist(
                        self.filledImage, self.contours, bx, by, bw, bh)
            w = self.weigthing_function
            xP = pixellist_xcoord
            yP = pixellist_ycoord
            wCx = [np.sum(x*w(self.img[y,x])) / np.sum(w(self.img[y,x])) \
                  if np.sum(w(self.img[y,x])) > 0 else 0 \
                  for x, y in zip(xP, yP)]
            wCy = [np.sum(y*w(self.img[y,x])) / np.sum(w(self.img[y,x])) \
                  if np.sum(w(self.img[y,x])) > 0 else 0 \
                  for x, y in zip(xP, yP)]

            self.particles["cx"] = wCx
            self.particles["cy"] = wCy

        # intensities
        if("intensities" in self.flags):
            if bounding_box is None:
                bounding_box, bx, by, bw, bh = calcBoundingBox(self.contours)
            if pixellist_xcoord is None or pixellist_xcoord is None:
                if self.filledImage is None:
                    self.getFilledImage()
                pixellist_ycoord, pixellist_xcoord = calcPixellist(
                        self.filledImage, self.contours, bx, by, bw, bh)
            w = self.weigthing_function
            xP = pixellist_xcoord
            yP = pixellist_ycoord
            minI = [np.quantile(self.img[y,x], 0.1) if len(x)>0 else 0
                    for x, y in zip(xP, yP)]
            maxI = [np.quantile(self.img[y,x], 0.9) if len(x)>0 else 0
                    for x, y in zip(xP, yP)]

            self.particles["minI"] = minI
            self.particles["maxI"] = maxI


        # Distance transform
        if("dist" in self.flags):
            if bounding_box is None:
                bounding_box, bx, by, bw, bh = calcBoundingBox(self.contours)
            if pixellist_xcoord is None or pixellist_xcoord is None:
                if self.filledImage is None:
                    self.getFilledImage()
                pixellist_ycoord, pixellist_xcoord = calcPixellist(
                        self.filledImage, self.contours, bx, by, bw, bh)
            self.getDistImage()
            dist = []
            for i in range(len(self.contours)):
                distImage = self.distImage[by[i]:by[i]+bh[i], \
                                           bx[i]:bx[i]+bw[i]].copy()
                pY = pixellist_ycoord[i] - by[i]
                pX = pixellist_xcoord[i] - bx[i]
                dist.append(np.quantile(distImage[pY, pX], 0.975))
                """if area[i] > 100:
                    mask = np.zeros(distImage.shape).astype(np.uint8)
                    mask[pY, pX] = [255]
                    distImage = cv2.bitwise_and(distImage, distImage, mask=mask)
                    plt.figure()
                    plt.imshow(distImage)
                    plt.show()"""
            self.particles["dist"] = dist


        # Eigenvectors
        if("eigen" in self.flags):
            ew1, ew2 = calcEigenvalues(self.contours)
            self.particles['ew1'] = ew1
            self.particles['ew2'] = ew2


    def calcRegionpropsEmpty(self):
        flags = self.flags.copy()
        if "centroid" in flags:
            flags.remove("centroid")
            flags.append("cx")
            flags.append("cy")
        if "weighted_centroid" in flags:
            flags.remove("weighted_centroid")
            flags.append("cx")
            flags.append("cy")
        if "bounding_box" in flags:
            flags.remove("bounding_box")
            flags.append("bx")
            flags.append("by")
            flags.append("bw")
            flags.append("bh")
        if "pixellist" in flags:
            flags.remove("pixellist")
            flags.append("pixellist_xcoord")
            flags.append("pixellist_ycoord")
        if "extremes" in flags:
            flags.remove("extremes")
            flags.append("leftmost")
            flags.append("rightmost")
            flags.append("topmost")
            flags.append("bottommost")
        if "intensities" in flags:
            flags.remove("intensities")
            flags.append("minI")
            flags.append("maxI")
        if "eigen" in flags:
            flags.remove("eigen")
            flags.append("ew1")
            flags.append("ew2")

        for flag in flags:
            self.particles[flag] = []
