
import cv2
import numpy as np
import numbers
import json
import math
import time

def normalize(an_array):
    norm = np.linalg.norm(an_array)
    normal_array = an_array/norm
    return normal_array

def getAngleBetweenVectors2D(v1, v2):
    numerator = v1[0]*v1[1] + v2[0]*v2[1]
    denominator = np.sqrt(np.power(v1[0],2) + np.power(v2[0],2)) * np.sqrt(np.power(v1[1],2) + np.power(v2[1],2))
    result = np.arccos(numerator/denominator)
    return result

def getVec(v1, v2):
    res = [v1[0] - v2[0], v1[1] - v2[1]]
    resN = normalize(res)
    return resN

def isInInterval(y):
    isTrue = (  (np.pi/4 > y > 0) or
                (np.pi*0.75 < y < np.pi*1.25) or
                (-np.pi/4 < y < 0) or
                (-np.pi*0.75 > y > -np.pi*1.25))
    return isTrue

def getAxis(y):
    if isInInterval(y):
        return [1,0]
    else:
        return [0,1]

def getAngle(x, c, axis):
    vd = axis
    vec = getVec(c, x)
    angle = getAngleBetweenVectors2D(vd, vec)*180/np.pi
    return angle

def isThereThisPoint(cam, w, yaw):
    axis = getAxis(yaw)
    c = [cam[0], cam[1]]
    if axis[0] == 1:
        angles = float('inf')
    elif axis[0] == 0:
        angles = float('-inf')
    for x in w:
        angle = getAngle(x, c, axis)
        if axis[0] == 1:
            if (angle < angles):
                angles = angle
            else:
                return False
        if axis[0] == 0:
            if (angle > angles):
                angles = angle
            else:
                return False
    if (angles != None):
        return True

def sortByImg(a):
    return a[1][0]

def sortCoordByImg(w, p):
    coord_list = []
    for i in range (1, len(w)+1):
        coord = [w[i-1], p[i-1], i-1]
        coord_list.append(coord)
    coord_list.sort(key=sortByImg)
    obj = []
    img = []
    index = []
    for x in coord_list:
        obj.append(x[0])
        img.append(x[1])
        index.append(x[2])
    return [obj, img, index]

def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def roundMatrices(m):
    mat_arr = [ m.item(0,0), m.item(0,1), m.item(0,2),
                m.item(1,0), m.item(1,1), m.item(1,2),
                m.item(2,0), m.item(2,1), m.item(2,2)]
    m_arr = []
    for e in mat_arr:
        if e < 0.00001 and e > 0:
            m_arr.append(round_down(e, 4))
        elif e < 0 and e > -0.00001:
            m_arr.append(round_up(e, 4))
        elif e == -0:
            e = 0
            m_arr.append(e)
        else:
            m_arr.append(e)

    rounded_matrix = np.matrix([[m_arr[0], m_arr[1], m_arr[2]],
                                [m_arr[3], m_arr[4], m_arr[5]],
                                [m_arr[6], m_arr[7], m_arr[8]]])
    return rounded_matrix

def getYprFromRotationMatrix(r):
    roll = math.atan2(-r.item(2,0), math.sqrt(math.pow(r.item(0,0), 2) + math.pow(r.item(1,0), 2))); # beta
    cosRoll = math.cos(roll)
    yaw   = math.atan2(r.item(1,0) / cosRoll, r.item(0,0) / cosRoll)
    pitch  = math.atan2(r.item(2,1) / cosRoll, r.item(2,2) / cosRoll)
    return [yaw, pitch, roll]

def getRotationMatrixFromYpr(ypr):
    y = ypr[0]
    p = ypr[1]
    r = ypr[2]

    sinA = math.sin(y)
    cosA = math.cos(y)
    sinB = math.sin(r)
    cosB = math.cos(r)
    sinC = math.sin(p)
    cosC = math.cos(p)

    rotation_matrix = np.float32([
                            [cosA*cosB, cosA*sinB*sinC - sinA*cosC, cosA*sinB*cosC + sinA*sinC],
                            [sinA*cosB, sinA*sinB*sinC + cosA*cosC, sinA*sinB*cosC - cosA*sinC],
                            [-sinB, cosB*sinC, cosB*cosC]])
    return rotation_matrix

def getFOV(focal_length, sensor_size):
    s  = 18
    fov = 2 * np.arctan(s/focal_length)
    return fov

def getCameraMatrix(f, sensor_y, imgSize, isDefault):
    cx = imgSize[0]/2
    cy = imgSize[1]/2
    sensor_x1 = 36
    sensor_y1 = imgSize[1]/(imgSize[0]/36)

    fx = (cx*2 * f)/sensor_x1
    fy = (cy*2 * f)/sensor_y1

    camera_matrix = np.float32([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])
    return camera_matrix

def transformCoords(w):
    new_w = []
    for c in w:
        transfered = [c[0], -c[2], c[1]]
        new_w.append(transfered)
    return new_w

def sortByIndex(a):
    return a[3]

def getAccuracyErrors(input_pts, project_pts, indexes):
    dist_arr = []
    global_inaccuracy = 0
    for x in range(0, len(input_pts)):
        x_pair = abs(input_pts[x][0] - project_pts[x][0])
        y_pair = abs(input_pts[x][1] - project_pts[x][1])
        dist = math.sqrt(math.pow(x_pair, 2) + math.pow(y_pair, 2))
        global_inaccuracy += dist
        dist_arr.append([x_pair, y_pair, dist, indexes[x]])

    dist_arr.sort(key=sortByIndex)

    dArr = []
    for i in dist_arr:
        dArr.append({
            "dx": i[0],
            "dy": i[1],
            "dl": i[2]
        })
    return [dArr, global_inaccuracy]

def getTvec(r, cam_pose):
    r = np.matrix(r)
    cam_pose = np.matrix([
                            [cam_pose[0]],
                            [cam_pose[1]],
                            [cam_pose[2]]])

    negative_rotation = -r
    transposed = negative_rotation.T
    inversed = np.linalg.inv(transposed)
    position = inversed * cam_pose
    result = np.float32(fourth_step)
    return result

def getUsableImagePoints(im):
    new_im_arr = []
    for i in im:
        point = [i.item(0,0), i.item(0,1)]
        new_im_arr.append(point)
    return new_im_arr

def getProjectPoints(w, cam_position, camera_matrix, r_vec, t_vec):
    image_points, jacobian = cv2.projectPoints(w, r_vec, t_vec, camera_matrix, None)
    result = getUsableImagePoints(image_points)
    return result

def getCropFactor(s1, s2):
    diagonal1 = math.sqrt(s1[0] ** 2 + s1[1] ** 2)
    diagonal2 = math.sqrt(s2[0] ** 2 + s2[1] ** 2)
    result = diagonal2/diagonal1
    return result

def get35SensorFromImgSize(imgSize):
    k = imgSize[0]/36
    sensorX = 36
    sensorY = imgSize[1]/k
    return [sensorX, sensorY]

def normalizeFS(f, s, imgSize):
    sizeFactor = imgSize[0]/imgSize[1]
    k = imgSize[0]/36
    sensorX = 36
    sensorY = imgSize[1]/k
    cropFactor = getCropFactor([s * sizeFactor, s], [sensorX, sensorY])
    focal = f * cropFactor
    fov2 = getFOV(focal, sensorY)
    return [focal, [sensorX, sensorY]]

def solvePnP(w, p, imgSize, f, s, iterations, indexes, isDefault):
    ite = int(iterations)
    object_points = np.float32(transformCoords(w))
    image_points = np.float32(p)
    camera_matrix = getCameraMatrix(f, s, imgSize, isDefault)
    if (len(object_points) == 3):
        error = 0
        try:
            retval, rvecs, tvecs = cv2.solveP3P(object_points,
                                                image_points,
                                                camera_matrix,
                                                None,
                                                cv2.SOLVEPNP_P3P)
        except:
            error = 1

        if error == 1:
            return False
        elif len(rvecs) < 1:
            return False

        rvec = rvecs[0]
        tvec = tvecs[0]
        r_vector = np.float32([
                    [rvec.item(0,0)],
                    [-rvec.item(2,0)],
                    [rvec.item(1,0)]])

    elif (len(object_points) > 3):
        error = 0
        try:
            retval, rvec, tvec, inliers	= cv2.solvePnPRansac(object_points,
                                                             image_points,
                                                             camera_matrix,
                                                             None,
                                                             useExtrinsicGuess = False,
                                                             iterationsCount = ite,
                                                             reprojectionError = 0.0,
                                                             confidence = 0.99,
                                                             flags = cv2.SOLVEPNP_ITERATIVE)
        except:
            error = 1

        if error == 1:
            return False
        elif len(rvec) < 1:
            return False

        r_vector = np.float32([
                        [rvec[0]],
                        [-rvec[2]],
                        [rvec[1]]])

    rotationDef, jacobianDef = cv2.Rodrigues(rvec)
    rotation, jacobian = cv2.Rodrigues(r_vector)
    rounded_rotation = roundMatrices(rotation)
    ypr = getYprFromRotationMatrix(rounded_rotation)
    camera_position = -np.matrix(rotationDef).T * np.matrix(tvec)
    cp_transferedBack = [   camera_position.item(0,0),
                            camera_position.item(2,0),
                            -camera_position.item(1,0)]
    project_pts = getProjectPoints(object_points, camera_position, camera_matrix, rvec, tvec)
    if len(w) > 3:
        accuracyErrors = getAccuracyErrors(p, project_pts, indexes)
    else:
        accuracyErrors = [None, None]
    return [cp_transferedBack, ypr, accuracyErrors[0], accuracyErrors[1], f, s]

def solve_F1S1(w, p, size, foc, iterations, indexes, sortingMethod):
    result_list = []
    xmin = float('inf')
    xmax = float('-inf')
    ymin = float('inf')
    ymax = float('-inf')
    s = get35SensorFromImgSize(size)[1]

    for i in range(1, 1000):
        if i < 500:
            f = foc - (i / 1000)
        elif i > 500:
            f = foc + (i - 500)/1000

        result = solvePnP(w, p, size, f, s, iterations, indexes, False)
        if result == False:
            isThere = False
        else:
            isThere = isThereThisPoint(result[0], w, result[1][0])
        if isThere == True:
            result_list.append(result)
            xmax = max(xmax, result[0][0])
            xmin = min(xmin, result[0][0])
            ymax = max(ymax, result[0][1])
            ymin = min(ymin, result[0][1])

    if len(result_list) > 1:
        if sortingMethod:
            result_list.sort(key=sortByInaccuracy)
            center_point = result_list[0]

        else:
            if (xmax - xmin) > (ymax - ymin):
                result_list.sort(key=sortByX)
            elif (xmax - xmin) < (ymax - ymin):
                result_list.sort(key=sortByY)

            median = int(len(result_list)/2)
            center_point = result_list[median-1]

        return  {
                    'point': center_point[0],
                    'ypr': center_point[1],
                    'points_inaccuracy': center_point[2],
                    'global_inaccuracy': center_point[3],
                    #'details': {'result_list': result_list},
                    'focalLength': center_point[4],
                    'sensorSize': [36, center_point[5]],
                }

    if len(result_list) == 1:
        return  {
                    'point': result_list[0][0],
                    'ypr': result_list[0][1],
                    'points_inaccuracy': center_point[0][2],
                    'global_inaccuracy': center_point[0][3],
                    #'details': {'result_list': result_list},
                    'focalLength': result_list[0][4],
                    'sensorSize': [36, result_list[0][5]]
                }

def solve_F1S0(w, p, size, f, sortingMethod, iterations, indexes):
    result_list = []
    xmin = float('inf')
    xmax = float('-inf')
    ymin = float('inf')
    ymax = float('-inf')
    for i in range(1, 2000):
        s = 2 + 0.1 * i
        fov = getFOV(f, s)
        if 0.7 < fov < 1.3:
            result = solvePnP(w, p, size, f, s, iterations, indexes, False)
            if result == False:
                isThere = False
            else:
                isThere = isThereThisPoint(result[0], w, result[1][0])
            if isThere == True:
                result_list.append(result)
                xmax = max(xmax, result[0][0])
                xmin = min(xmin, result[0][0])
                ymax = max(ymax, result[0][1])
                ymin = min(ymin, result[0][1])

    if len(result_list) > 1:
        if sortingMethod:
            result_list.sort(key=sortByInaccuracy)
            center_point = result_list[0]
        else:
            if (xmax - xmin) > (ymax - ymin):
                result_list.sort(key=sortByX)
            elif (xmax - xmin) < (ymax - ymin):
                result_list.sort(key=sortByY)
            median = int(len(result_list)/2)
            center_point = result_list[median-1]

        normalized = normalizeFS(center_point[4], center_point[5], size)
        return  {
                    'point': center_point[0],
                    'ypr': center_point[1],
                    'points_inaccuracy': center_point[2],
                    'global_inaccuracy': center_point[3],
                    #'details': {'result_list': result_list},
                    'focalLength': normalized[0],
                    'sensorSize': normalized[1],
                }

    if len(result_list) == 1:
        normalized = normalizeFS(result_list[0][4], result_list[0][5], size)
        return  {
                    'point': result_list[0][0],
                    'ypr': result_list[0][1],
                    'points_inaccuracy': result_list[0][2],
                    'global_inaccuracy': result_list[0][3],
                    #'details': {'result_list': result_list},
                    'focalLength': normalized[0],
                    'sensorSize': normalized[1],
                }


def solve_F0S1(w, p, size, s, sortingMethod, iterations, indexes):
    result_list = []
    xmin = float('inf')
    xmax = float('-inf')
    ymin = float('inf')
    ymax = float('-inf')
    for i in range(1, 2000):
        f = 2 + 0.1 * i
        fov = getFOV(f, s)
        if 0.7 < fov < 1.3:
            result = solvePnP(w, p, size, f, s, iterations, indexes, False)
            if result == False:
                isThere = False
            else:
               isThere = isThereThisPoint(result[0], w, result[1][0])
            if isThere == True:
                result_list.append(result)
                xmax = max(xmax, result[0][0])
                xmin = min(xmin, result[0][0])
                ymax = max(ymax, result[0][1])
                ymin = min(ymin, result[0][1])

    if len(result_list) > 1:
        if sortingMethod:
            result_list.sort(key=sortByInaccuracy)
            center_point = result_list[0]
        else:
            if (xmax - xmin) > (ymax - ymin):
                result_list.sort(key=sortByX)
            elif (xmax - xmin) < (ymax - ymin):
                result_list.sort(key=sortByY)
            median = int(len(result_list)/2)
            center_point = result_list[median-1]

        normalized = normalizeFS(center_point[4], center_point[5], size)
        return  {
                    'point': center_point[0],
                    'ypr': center_point[1],
                    'points_inaccuracy': center_point[2],
                    'global_inaccuracy': center_point[3],
                    #'details': {'result_list': result_list},
                    'focalLength': normalized[0],
                    'sensorSize': normalized[1],
                }

    if len(result_list) == 1:
        normalized = normalizeFS(result_list[0][4], result_list[0][5], size)
        return  {
                    'point': result_list[0][0],
                    'ypr': result_list[0][1],
                    'points_inaccuracy': result_list[0][2],
                    'global_inaccuracy': result_list[0][3],
                    #'details': {'result_list': result_list},
                    'focalLength': normalized[0],
                    'sensorSize': normalized[1],
                }

def sortByX(a):
    return a[0][0]

def sortByInaccuracy(a):
    return a[3]

def sortByY(a):
    return a[0][1]

def solve_F0S0(w, p, size, sortingMethod, iterations, indexes):
    result_list = []
    xmin = float('inf')
    xmax = float('-inf')
    ymin = float('inf')
    ymax = float('-inf')
    s = get35SensorFromImgSize(size)[1]

    for j in range(1, 20000):
        f = 2 + 0.01 * j
        fov = getFOV(f, s)
        if 0.7 < fov < 1.3:
            result = solvePnP(w, p, size, f, s, iterations, indexes, True)
            if result == False:
                isThere = False
            else:
                isThere = isThereThisPoint(result[0], w, result[1][0])
            if isThere == True:
                result_list.append(result)
                xmax = max(xmax, result[0][0])
                xmin = min(xmin, result[0][0])
                ymax = max(ymax, result[0][1])
                ymin = min(ymin, result[0][1])
    if len(result_list) > 1:
        if sortingMethod:
            result_list.sort(key=sortByInaccuracy)
            center_point = result_list[0]

        else:
            if (xmax - xmin) > (ymax - ymin):
                result_list.sort(key=sortByX)
            elif (xmax - xmin) < (ymax - ymin):
                result_list.sort(key=sortByY)

            median = int(len(result_list)/2)
            center_point = result_list[median-1]

        return {
                    'point': center_point[0],
                    'ypr': center_point[1],
                    'points_inaccuracy': center_point[2],
                    'global_inaccuracy': center_point[3],
                    #'details': {'result_list': result_list},
                    'focalLength': center_point[4],
                    'sensorSize': [36, center_point[5]]
                }

    if len(result_list) == 1:
        return {
                    'point': result_list[0][0],
                    'ypr': result_list[0][1],
                    'points_inaccuracy': result_list[0][2],
                    'global_inaccuracy': result_list[0][3],
                    #'details': {'result_list': result_list},
                    'focalLength': result_list[0][4],
                    'sensorSize': [36, result_list[0][5]]
                }

def checkF(f):
    problem = []
    if f != None:
        if not isinstance(f, numbers.Number):
            problem.append('Focal length must be number')
        if isinstance(f, list):
            problem.append('Focal length must be single number')
        elif len(problem) == 0:
            if f < 0:
                problem.append('Focal length must be positive number')

    return problem

def checkS(s):
    problem = []
    if s != None:
        if isinstance(s, list):
            if len(s) != 2:
                problem.append('Sensor size array must contain 2 values')
            for i in s:
                if not isinstance(i, numbers.Number):
                    problem.append('Sensor size must be number')
                else:
                    if i < 0:
                        problem.append('Sensor size must be positive number')
        else:
            problem.append('Sensor size must be array or null')

    return problem

def checkImgSize(size):
    problem = []
    if size != None:
        if isinstance(size, list):
            if len(size) != 2:
                problem.append('Image size array must contain 2 values')
            for i in size:
                if not isinstance(i, numbers.Number):
                    problem.append('Image size must be number')
                else:
                    if i < 0:
                        problem.append('Image size must be positive number')
        else:
            problem.append('Image size must be array or null')
    else:
        problem.append('Image size must be set')
    return problem

def sortCheck(a):
    return a

def checkListLength(x, l):
    arr = []
    for i in x:
        if len(i) != l:
            arr.append(0)
        else:
            arr.append(1)
    arr.sort(key=sortCheck)
    if arr[0] == 0:
        return False
    else:
        return True

def checkWP(w, p):
    problem = []
    for a in w:
        for k in a:
            if not isinstance(k, numbers.Number):
                problem.append('World coords must be numbers')
                break
        else:
            continue
        break

    for b in p:
        for l in b:
            if not isinstance(l, numbers.Number):
                problem.append('Image coords must be numbers')
                break
        else:
            continue
        break

    if len(problem) == 0:
        if (len(w) < 3):
            problem.append('Minimum number of world points is 3')
        else:
            if not checkListLength(w, 3):
                problem.append('X (world points) each array must contain 3 values')

        if (len(p) < 3):
            problem.append('Minimum number of image points is 3')
        else:
            if not checkListLength(p, 2):
                problem.append('U (image points) each array must contain 2 values')
            else:
                if p[0][0] == p[1][0] or p[0][0] == p[2][0] or p[1][0] == p[2][0]:
                    problem.append('Image points must have different x coord')
        if len(w) != len(p):
            problem.append('Number of X (world points) and U (image points) must equal')

    return problem

def checkIterations(i):
    problem = []
    if not isinstance(i, numbers.Number):
        problem.append('Iteration must be number')
    if isinstance(i, list):
        problem.append('Iteration must be single number')
    elif len(problem) == 0:
        if i < 10:
            problem.append('Number of iterations must be > 10')

    return problem


def checkRequirements(w, p, size, f, s, i):
    p1 = checkF(f)
    p2 = checkS(s)
    p3 = checkImgSize(size)
    p4 = checkWP(w, p)
    p5 = checkIterations(i)
    return p1 + p2 + p3 + p4 + p5

def getSortingMethod(w):
    if (len(w) == 3 or len(w) == 4):
        return False
    else:
        return True

def solver(object_points, image_points, image_size, focal_length, sensor_size, iterations):
    if iterations == None:
        iterations = 10

    errorList = checkRequirements(  object_points,
                                    image_points,
                                    image_size,
                                    focal_length,
                                    sensor_size,
                                    iterations)

    sortingMethod = getSortingMethod(object_points)
    if len(errorList) == 0:
        points = sortCoordByImg(object_points, image_points)
        object_points = points[0]
        image_points = points[1]
        index_list = points[2]
        result = None
        if focal_length:
            result = solve_F1S1(object_points,
                                image_points,
                                image_size,
                                focal_length,
                                iterations,
                                index_list,
                                sortingMethod)
        else:
            result = solve_F0S0(object_points,
                                image_points,
                                image_size,
                                sortingMethod,
                                iterations,
                                index_list)

        if (result != None):
            return result
        else:
            return {'errors': 'It cannot be solved'}
    else:
        return {'errors': errorList}
