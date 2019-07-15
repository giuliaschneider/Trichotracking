def calc_chamber(background):
    """ Get bw image, pixels inside set to 255, outside 0. """
    bw = cv2.adaptiveThreshold(background, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 41, 8)
    # Close gaps by morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    h, w = bw.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(bw, mask, (int(w/2),int(h/2)), 255)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return bw


def calc_chamber_df_ulisetup(background):
    """ Get bw image, pixels inside set to 255, outside 0,
        adjusted for uli's setup """
    # Find well wall (high intensity circle)
    ret, bw = cv2.threshold(background,200,255,cv2.THRESH_BINARY)
    h, w = bw.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    set_trace()
    cv2.floodFill(bw, mask, (int(w/2),int(h/2)), 255)
    mask[mask==1] = [255]
    return mask[1:-1,1:-1]
