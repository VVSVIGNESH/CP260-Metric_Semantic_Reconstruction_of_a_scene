def __init__(
    self,
    point_cloud,
    images,
    poses,
    K
):

    self.point_cloud = point_cloud

    self.images = images

    self.poses = poses

    self.K = K

    self.geometry = GeometryProcessor()

    # ORB tuned for small metallic connectors
    self.detector = cv2.ORB_create(

        nfeatures=8000,

        scaleFactor=1.2,

        nlevels=8,

        edgeThreshold=5,

        firstLevel=0,

        WTA_K=2,

        scoreType=cv2.ORB_HARRIS_SCORE,

        patchSize=31,

        fastThreshold=5
    )

    # brute-force Hamming matcher
    self.matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    # physically realistic socket size priors
    # [min_extent, max_extent]
    self.size_priors = {

        'power_socket': [0.04, 0.08],

        'ethernet_socket': [0.02, 0.05],

        'vga_socket': [0.03, 0.08],

        'hdmi_socket_left': [0.01, 0.03],

        'usb_socket_top_right': [0.01, 0.03]
    }