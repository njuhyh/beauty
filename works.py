from copy import deepcopy, copy
import cv2
import os
import mediapipe as mp
import numpy as np
import math
from scipy.spatial.transform import Rotation
from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters
from panda3d.core import OrthographicLens, WindowProperties, ColorWriteAttrib, FrameBufferProperties, WindowProperties, GraphicsPipe, AmbientLight, PointLight, AntialiasAttrib, loadPrcFileData

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

loadPrcFileData('','framebuffer-multisample 1')
loadPrcFileData('','multisamples 8')

# added models, for accessries
SUPPORTED_MODELS = {
    'glasses': 'models/aligned_glasses.gltf',
    'crown': 'models/crown.gltf',
    'chopper': 'models/chopper.gltf'
}
# models for rendering
MODELS = {
    'face': 'models/canonical_face_model.gltf',
}

# indices of vertices in the contour of face mesh
FACE_POLY = np.array([
    21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162
])
EYEPOLY1 = np.array([
    27, 29, 30, 247, 226, 110, 24, 23, 22, 26, 112, 243, 190, 56, 28
])
EYEPOLY2 = np.array([
    257, 259, 260, 467, 446, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258
])
LIPSPOLY1 = np.array([
    0, 17, 84, 181, 91, 146, 61, 37
])
LIPSPOLY2 = np.array([
    0, 17, 314, 405, 321, 375, 291, 267
])

# canonical face model reference point coordination
TOP = np.array([0, -4.4815, 8.2618])
TOP_LEFT_7 = np.array([-7.7431, 2.0052, 2.365])
TOP_RIGHT_7 = np.array([7.7431, 2.0052, 2.365])
EYE_CENTER = np.array([0, -5.7886, 2.4733])
MODEL_REF = np.array([TOP, TOP_LEFT_7, TOP_RIGHT_7, EYE_CENTER])

# face mesh reference point indices for model, top, top_left, top_right, eye_center
FACE_MODEL_REF = np.array([10, 127, 356, 6])

def to_xy_shape(shape):
    return np.array([shape[1], shape[0]])

def xOy_coor(vec, shape):
    return np.array(vec[:2] * shape).astype(dtype=np.uint32)

def mark_vec(img, vec, size, color):
    cv2.circle(
        img,
        xOy_coor(vec, to_xy_shape(img.shape)),
        size,
        color,
        cv2.FILLED
    )

def create_base():
    w, h = 1437, 1920
    base = ShowBase(windowType='offscreen')
    base.buffer = base.graphicsEngine.makeOutput(base.pipe, 'buffer', -1, FrameBufferProperties.getDefault(), WindowProperties(size=(w,h)), GraphicsPipe.BFFbPropsOptional | GraphicsPipe.BFRefuseWindow | GraphicsPipe.BFResizeable)
    dr = base.buffer.makeDisplayRegion()
    dr.setCamera(base.cam)
    base.buffer.setClearColor((0, 0, 0, 0))
    base.render.setAntialias(AntialiasAttrib.MMultisample)
    return base

def create_mesh_generator():
    return mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)

'''
    Data format transformation functions
'''
def mp_marks_all_exists(landmarks):
    for mark in iter(landmarks):
        if mark.x < 0 or mark.x >= 1 or mark.y < 0 or mark.y >=1:
            return False
    return True

def normalize_coordinate(ratio_x, ratio_y, x, y):
    return ratio_x*x, ratio_y*y

def normalize_coordinate_int(ratio_x, ratio_y, x, y):
    return round(ratio_x*x), round(ratio_y*y)

# convert mp landmark objects to array
def mp_marks_to_cv(landmarks, x, y, indices=None, is_int=False):
    marks = []
    norm = normalize_coordinate_int if is_int else normalize_coordinate
    if indices is not None:
        for i in iter(indices):
            m = landmarks[i]
            pt = norm(m.x, m.y, x, y)
            marks.append(pt)
    else:
        for mark in iter(landmarks):
            pt = norm(mark.x, mark.y, x, y)
            marks.append(pt)
    marks = np.array(marks)
    return marks

# convert mp landmark objects to 3d vectors
def mp_marks_to_vector(landmarks, indices = None):
    at_indices = np.array(landmarks, dtype=object)[np.array(indices, dtype=np.uint64)]
    res = [[e.x,e.y,e.z] for e in at_indices]
    return np.array(res)

# # select specific marks in mesh
# def get_specific_marks(face_landmarks, indices):
#     return mp_marks_to_cv(indices)

# store a image
def store_img(img, name):
    cv2.imwrite(os.getcwd() + '\\tmp\\' + name + '.png', img)
    
# transform image xyz to cv xyz
# original xy are ratios to the width and height
def xyz_cv_to_render(vec, shape):
    ratio_yx = shape[1]*1.0/shape[0]
    return np.array([vec[0]-0.5, vec[2], (0.5-vec[1])*ratio_yx])

# tranform render xyz to cv xyz
def xyz_render_to_cv(vec, shape):
    ratio_xy = shape[0]*1.0/shape[1]
    return np.array([vec[0]+0.5, 0.5-vec[2]*ratio_xy, vec[1]])


'''
    Basic image processing functions
'''
# calculate source point in local transition algorithm
def local_transition_source(p, c, t, r):
    '''
    Args
    ------
    x, y : The destination point
    cx, cy : The transition center
    tx, ty : The transition target center
    r : The transition radius
    '''
    rr = r*r
    dd = np.sum((p-c)**2)
    # if dd >= rr:
    #     return x, y
    tt = np.sum((t-c)**2)
    ratio = (1-tt/(rr-dd+tt))**2
    src = p - (t-c)*ratio
    return src

# calculate source point in poly single core transition algorithm
def poly_transition_source(p, c, c_, v_list):
    # First, check p' area
    len = v_list.shape[0]
    dv_list = v_list - c_
    v_direction = np.sign(np.cross(dv_list[0], dv_list[1]))
    dp = p - c_
    cross_list = np.cross(dv_list, dp)
    area_num = 0
    for i, cro in cross_list:
        if np.sign(cross_list[i+1]) == v_direction and np.sign(cro) != v_direction:
            area_num = i
    area_next = (area_num+1) % len
    T_dv = np.linalg.inv(np.array([dv_list[area_num], dv_list[area_next]]).T)
    vp = np.matmul(T_dv, dp)
    return np.matmul((np.array([v_list[area_num],v_list[area_next]]) - c).T, vp) + c

# run poly transition
def run_poly_transition(img, res_img, c, c_, v_list):
    h, w = img.shape[:2]
    len = v_list.shape[0]
    dv_list = v_list - c_
    T_list = np.ndarray((len, 2, 2), dtype=np.float64)
    for i in range(len):
        i_ = (i+1)%len
        if np.cross(dv_list[i], dv_list[i_]) == 0:
            dv_list[i_] += dv_list[i_][::-1] * np.array([-1.0, 1.0]) * 0.01
        T_list[i] = np.linalg.inv(np.array([dv_list[i], dv_list[i_]]).T)
        origin_base = (np.array([v_list[i],v_list[i_]]) - c).T

        # calculate range in ints
        tri_ = np.array([c_, v_list[i], v_list[i_]])
        tmin = tri_.min(axis=0)
        tmax = tri_.max(axis=0)
        ymin = max(math.floor(tmin[1]), 0)
        ymax = min(math.ceil(tmax[1])+1, h)
        xmin = max(math.floor(tmin[0]), 0)
        xmax = min(math.ceil(tmax[0]), w)

        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                p = np.array([x,y], dtype=np.float64)
                dp = p - c_
                vp = np.matmul(T_list[i], dp)
                v0, v1 = vp
                if v0 > 0.0 and v0 <= 1.0 and v1 >= 0.0 and v1 <= 1.0 and v0+v1 <= 1:
                    res_img[y, x] = bil_interpolation(img, np.matmul(origin_base, vp)+c)

# run local transition (no range check)
# def run_local_transition(img, res_img, c, t, r):
#     rr = r*r
#     for y in range(math.floor(c[1]-r), math.ceil(c[1]+r)+1):
#         for x in range(math.floor(c[0]-r), math.ceil(c[0]+r)+1):
#             p = np.array([x, y], dtype=np.float64)
#             if np.sum((p-c)**2) <= rr:
#                 res_img[y, x] = bil_interpolation(img, local_transition_source(p, c, t, r))


# bilinear interpolation
def bil_interpolation(img, p):
    '''
    Args
    ------
    img : data source
    x, y : coordinate in float
    '''
    x = p[0]
    y = p[1]
    x0 = math.floor(x)
    x1 = x0 + 1
    y0 = math.floor(y)
    y1 = y0 + 1
    if x1>=img.shape[1] or y1>=img.shape[0]:
        return img[-1, -1]
    w00 = img[y0, x0].astype(np.float64) * (x1-x) * (y1-y)
    w01 = img[y1, x0].astype(np.float64) * (x1-x) * (y-y0)
    w10 = img[y0, x1].astype(np.float64) * (x-x0) * (y1-y)
    w11 = img[y1, x1].astype(np.float64) * (x-x0) * (y-y0)
    return  (w00 + w01 + w10 + w11).astype(np.uint8)

# cover img0 with alpha channel (img1 not) onto img1 
def alpha_cover(img0: cv2.Mat, img1: cv2.Mat):
    img = img0[:,:,:3] # exclude alpha channel
    img_alpha = img0[:,:,3]
    mask = img_alpha / 255.0
    mask = np.repeat(mask[:,:,None], 3, axis=2)
    inv_mask = 1.0 - mask
    result = (img * mask + img1 * inv_mask).astype(np.uint8)
    return result

# copy img1 to img0 through a blurred mask
def blur_copy(img0: cv2.Mat, img1: cv2.Mat, mask, blur_r):
    mask = cv2.blur(mask, (blur_r, blur_r))
    inv_mask = cv2.bitwise_not(mask)
    res = np.array(img0 * (inv_mask / 255.0) + img1 * (mask / 255.0), dtype=np.uint8)
    return res

# apply function to an image within blurred mask
def mask_process(img: cv2.Mat, mask: cv2.Mat, func, blur_r):
    img_ = img.copy()
    img_ = func(img_)
    res = blur_copy(img, img_, mask, blur_r)
    return res

# generate selected area with polygon intersection and difference
def select_ploy(sel_poly, rem_poly, shape):
    area = np.zeros(shape, dtype=np.uint8)
    rem_area = np.zeros(shape, dtype=np.uint8)
    try:
        cv2.fillPoly(area, sel_poly, (255, 255, 255))
    except Exception:
        print('area wrong')
        print('sel_poly:\n', sel_poly)
    try:
        cv2.fillPoly(rem_area, rem_poly, (255, 255, 255))
    except Exception:
        print('rem area wrong')
        print('rem_poly:\n', rem_poly)
    res = area & (~rem_area)
    return res

# calculate translation distances and rotation angles from CANONICAL coordinates and MESH coordinates
def calculate_transform(canonical_points, mesh_points, shape):
    '''
    Return
    ------
    tuple (mx, my, mz, x_ag, y_ag, z_ag, lin)
    mx, my, mz : coordinate in image coordinate system
    x_ag, y_ag, z_ag : angles in degree of rotations around each axis counterclockwise
    lin : linearity ratio
    '''
    ct, cl, cr, cc = canonical_points # canonical top, left, right, center
    for i, j in enumerate(mesh_points):
        mesh_points[i] = xyz_cv_to_render(j, shape)
    mt, ml, mr, mc = mesh_points # mesh top, left, right, center
    M = np.linalg.inv(np.array([cr-cc, cl-cc, ct-cc]).T)
    co_ = np.matmul(M, -cc.T) # canonical origin coordination in new system
    mo = np.matmul(np.array([mr-mc, ml-mc, mt-mc]).T, co_) + mc # mesh origin coordination (predicted)
    
    b1 = mt - mc
    b2 = mc - mo
    b3 = b2 + b1 * (- b2[2] / b1[2]) # intersection of xOy and b1,b2 platform

    z_r_ag = - math.pi/2 - math.atan2(b3[1], b3[0])
    z_r = Rotation.from_rotvec([0, 0, z_r_ag]).as_matrix() # rotate b3 to negative y axis
    b2_ = np.matmul(z_r, b2) # rotate b2

    b2_xOz = b2_.copy() # b2 projection on xOz
    b2_xOz[1] = 0

    y_r_ag = - math.atan2(b2_xOz[0], b2_xOz[2])
    y_r = Rotation.from_rotvec([0, y_r_ag, 0]).as_matrix()
    b2__ = np.matmul(y_r, b2_) # rotate b2_ to yOz

    x_r_ag = math.atan2(cc[2], cc[1]) - math.atan2(b2__[2], b2__[1]) # rotate b2_ in yOz to the canonical angle

    lin = np.linalg.norm(ml-mr)/np.linalg.norm(cl-cr) # linerity

    mo = xyz_render_to_cv(mo, shape)
    mx, my, mz = mo

    return (mx, my, mz, 
            -x_r_ag/(2*math.pi)*360, -y_r_ag/(2*math.pi)*360, -z_r_ag/(2*math.pi)*360,
            lin) # reverse all the angles and transform them into degrees


'''
    Image processing algorithms
'''
# face whitening
def face_whitening(img, landmarks):
    h, w = img.shape[0], img.shape[1]
    whiten = lambda x:cv2.bilateralFilter(x, 10, 20, 10)
    face_poly = mp_marks_to_cv(landmarks, w, h, FACE_POLY, True)
    eye1_poly = mp_marks_to_cv(landmarks, w, h, EYEPOLY1, True)
    eye2_poly = mp_marks_to_cv(landmarks, w, h, EYEPOLY2, True)
    lips1_poly = mp_marks_to_cv(landmarks, w, h, LIPSPOLY1, True)
    lips2_poly = mp_marks_to_cv(landmarks, w, h, LIPSPOLY2, True)
    mask = select_ploy([face_poly], [eye1_poly, eye2_poly, lips1_poly, lips2_poly], img.shape)
    res_img = mask_process(img, mask, whiten, 5)
    return res_img


# indices for thinning, left_top, left_bot, left_move, right_top, right_bot, right_move
# FACE_THIN_REF = np.array([147, 136, 207, 376, 365, 427])
# face thinning (no range check)
# def face_thinning(img, landmarks, radius, force):
#     '''
#     Args
#     ------
#     img : Data source
#     landmarks : Landmarks in mediapipe face_mesh
#     radius : deformation radius
#     force : The ratio of transition target position, usually from 0.0 to 1.0
#     '''
#     h, w = img.shape[:2]
#     lt, lb, lm, rt, rb, rm = mp_marks_to_cv(landmarks, w, h, FACE_THIN_REF)
#     # left center, radius, target
#     lc = (lt+lb)/2
#     lr = np.linalg.norm(lt-lc) * radius
#     lta = lc + (lm-lc) * force
#     # right center, radius, target
#     rc = (rt+rb)/2
#     rr = np.linalg.norm(rt-rc) * radius
#     rta = rc + (rm-rc) * force

#     res_img = img.copy()
#     run_local_transition(img, res_img, lc, lta, lr)
#     run_local_transition(img, res_img, rc, rta, rr)
#     return res_img 

FACE_THIN_REF2 = np.array([116, 138, 150, 214, 345, 367, 379, 434])
def face_thinning(img, landmarks, force):
    h, w = img.shape[:2]
    l1, lc, l2, lm, r1, rc, r2, rm = mp_marks_to_cv(landmarks, w, h, FACE_THIN_REF2)
    lc_ = lc + (lm-lc) * force
    rc_ = rc + (rm-rc) * force
    l3 = np.array([l1[0], l2[1]])
    r3 = np.array([r1[0], r2[1]])
    res_img = img.copy()
    run_poly_transition(img, res_img, lc, lc_, np.array([l1, l2, l3]))
    run_poly_transition(img, res_img, rc, rc_, np.array([r1, r2, r3]))
    return res_img

# generate face mesh from a pre-defined generator (USUALLY from mediapipe)
def generate_mesh(face_mesh_generator, image):
    return face_mesh_generator.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def cover_alpha_img(alpha_img, src_img):
    '''
    Cover another img with alpha channel to the source img.
    '''
    img = alpha_img[:,:,:3] # exclude alpha channel
    img_alpha = alpha_img[:, :, 3]

    mask = img_alpha / 255.0
    inv_mask = 1.0 - mask

    result = np.ndarray(img.shape, dtype=np.uint8)
    for c in range(3):
        result[:,:,c] = (img[:,:,c] * mask).astype(np.uint8) + (src_img[:,:,c] * inv_mask).astype(np.uint8)

def draw_mesh(img, mesh_result):
    for face_landmarks in mesh_result.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())

class MeshMatchRenderer:
    '''
    A renderer that produces output according to the window size and the mesh.
    '''
    def __init__(self, base):
        '''
        'base' must include an offscreen buffer, called base.buffer.
        '''
        self.modelType = None
        self.modelCache = {}
        self.modelList = []     # record the models manually added
        self.base = base        # take over global base
        self.hasLight = False
        self.hasFilter = False
        self.shape = None

    # change buffer size and 
    def setSize(self, shape):
        if shape == self.shape:
            return
        self.shape = to_xy_shape(shape)
        self.base.buffer.setSize(self.shape[0], self.shape[1])
        lens = OrthographicLens()
        lens.setFilmSize(1.0, self.shape[1]/self.shape[0])
        base.cam.node().setLens(lens)

    def setScalePosHpr(self, node, scale, x, y, z, h, p, r):
        node.setScale(scale)
        node.setPos(x, y, z)
        node.setHpr(h, p, r)

    def loadModel(self, name):
        new_node = None
        # Using cache here. Filenames of the same model must be the same.
        if name in self.modelCache:
            new_node = deepcopy(self.modelCache[name])
        else:
            new_node = self.base.loader.loadModel(MODELS[name])
            self.modelCache[name] = deepcopy(new_node)
        return new_node

    def createModel(self, name, scale, x, y, z, h, p, r, is_occluder=False):
        new_node = self.loadModel(name)
        # Using cache here. Filenames of the same model must be the same.
        self.setScalePosHpr(new_node, scale, x, y, z, h, p, r)
        self.modelList.append(new_node)
        new_node.reparentTo(self.base.render)
        if is_occluder:
            new_node.setBin('background', 0)
            new_node.setDepthWrite(True)
            new_node.setAttrib(ColorWriteAttrib.make(ColorWriteAttrib.COff))
        return new_node

    def addFace(self, marks, accessory_name):
        ref_marks = mp_marks_to_vector(marks, FACE_MODEL_REF)
        yx_ratio = self.shape[1] / self.shape[0]
        mx, my, mz, x_ag, y_ag, z_ag, lin = calculate_transform(MODEL_REF, ref_marks, self.shape)
        x, y, z, h, p, r = mx-0.5, mz*100, (0.5-my)*yx_ratio, z_ag, x_ag, y_ag
        # Create the face along with the aligned model, using the same transformation.
        # Create face as occluder
        self.createModel('face', lin, x, y, z, h, p, r, True)
        # Create accessory and apply Antialias
        acc = self.createModel(accessory_name, lin, x, y, z, h, p, r)


    def clearModels(self):
        for model in self.modelList:
            model.detachNode()
        self.modelList.clear()

    def getRenderResult(self):
        '''
        Get rendering result, always from the buffer.
        '''
        self.base.graphicsEngine.renderFrame()
        tex = self.base.buffer.getScreenshot()
        data = tex.getRamImage()
        v = memoryview(data).tolist()
        img = np.array(v, dtype=np.uint8)
        img = img.reshape((tex.getYSize(), tex.getXSize(), 4))
        img = img[::-1]
        return img

    def addLight(self):
        # add ambient light
        alight = AmbientLight("alight")
        alight.setColor((0.9, 0.9, 0.9, 1))
        alight_node = self.base.render.attachNewNode(alight)
        self.base.render.setLight(alight_node)
        # add point light
        plight = PointLight('plight')
        plight.setAttenuation((5.0, 0.0, 0.0))
        plight_node = self.base.render.attachNewNode(plight)
        plight_node.setPos(0, -4, 1)
        self.base.render.setLight(plight_node)

    def addFilter(self):
        self.base.filters = CommonFilters(self.base.buffer, self.base.cam)
        self.base.filters.setCartoonInk(separation=0.6)

    def render(self, shape, mesh, accesory_name):
        # Clear existing nodes.
        self.clearModels()
        # Reset buffer size.
        self.setSize(shape)
        # Try adding lights and filters if they don't exist.
        if not self.hasLight:
            self.addLight()
            self.hasLight = True
        # if not self.hasFilter:
        #     self.addFilter()
        #     self.hasFilter = True
        # # Add faces and accessories according to meshes.
        for landmark_group in mesh.multi_face_landmarks:
            self.addFace(landmark_group.landmark, accesory_name)
        # Render and get result.
        return self.getRenderResult()

base = None
rd = None
face_mesh = None

def init():
    global base, rd, face_mesh
    base = create_base()
    rd = MeshMatchRenderer(base)
    face_mesh = create_mesh_generator()
    for path in MODELS.values():  
        assert os.path.isfile(path)
    MODELS.update(SUPPORTED_MODELS)

# generate accesory image from face mesh and image shape
def generate_accesory(mesh_result, shape, accessory_name):
    return rd.render(shape, mesh_result, accessory_name)

# generate accessoy image from image
# def generate_merge_image(img):
#     return generate_glasses_accesory