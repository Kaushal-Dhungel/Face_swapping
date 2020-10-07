import cv2 
import numpy as np 
import dlib 

def extract_index(np_array):
    index = None
    for num in np_array[0]:
        index = num
        break
    return index


detector = dlib.get_frontal_face_detector() # for face detection
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

frame = cv2.imread("c_w_2.jpg")
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

frame_2 = cv2.imread("scarlette_1.jpg")
# frame_2 = cv2.resize(frame_2,(frame.shape[0],frame.shape[1]))
gray_2 = cv2.cvtColor(frame_2,cv2.COLOR_BGR2GRAY)

frame_2_new_face = np.zeros_like(frame_2)

mask = np.zeros_like(gray)


faces = detector(gray)
for face in faces:

    #----------detect landmark------
    landmarks = predictor(gray,face)
    landmarks_points = []
    for i in range(0,68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        landmarks_points.append((x,y))
    
    points = np.array(landmarks_points, np.int32) # python list lai numpy array ma convert gareko
    
    # convex hull le points dinxa boundary ko
    convex_hull = cv2.convexHull(points)
    
    # # yasle boundary ko points ko base ma boundary korxa
    # cv2.polylines(frame,[convex_hull],True,(0,0,255),2)
    
    # now we want the mask of area of face, white colored
    cv2.fillConvexPoly(mask,convex_hull,255)

    # apply mask over the face
    face_img = cv2.bitwise_and(frame,frame,mask = mask)

    # since we have go the face, now we are going to draw the triangles over the face
    # Delaunay triangulation

    # find the rectangle of the face
    rect = cv2.boundingRect(convex_hull)
    
    subdiv = cv2.Subdiv2D(rect) # specify the rectangle wher you want triangles
    subdiv.insert(landmarks_points)  # insert the lanmark points on the subdiv
    triangles = subdiv.getTriangleList() # this gives the coordinates of all the triangles
    triangles = np.array(triangles,np.int32)

    indexes_triangles = []
    # loop over the coordinates
    for t in triangles:    # t has 6 points
        pt1 = (t[0],t[1])
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])


        # we can not apply different dilauney traingulation on face two cause this might draw triangles 
        # differently on the 2nd image. So we will use the coordinates of the first pic to draw the trangles 
        # on the second pic and later swap the triangle. For this what we will do is
        # find the index of landmark points based on the coordinates of 1st image and draw corresponding triangles on 2nd img 

        # to get the index of the points 
        index_pt1 = np.where((points == pt1).all(axis = 1))
        index_pt1 = extract_index(index_pt1)
        # print(index_pt1)  # this prints all the index of the  points 

        index_pt2 = np.where((points == pt2).all(axis = 1))
        index_pt2 = extract_index(index_pt2)
        
        index_pt3 = np.where((points == pt3).all(axis = 1))
        index_pt3 = extract_index(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1,index_pt2,index_pt3]
            indexes_triangles.append(triangle)


# for the 2nd image
faces_2 = detector(gray_2)
for face in faces_2:

    #----------detect landmark------
    landmarks = predictor(gray_2,face)
    landmarks_points_2 = []
    for i in range(0,68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        landmarks_points_2.append((x,y))

        # cv2.circle(frame_2,(x,y),3,(0,255,0),-1)

# drawing triangles on the 2nd image using the coordinates of the first face
for triangle_index in indexes_triangles:
    # print(triangle_index)  # triangle_index gives index of each points of the triangles of frame_1

    # for first img
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle_1 = np.array([tr1_pt1,tr1_pt2,tr1_pt3],np.int32)

    rect_1 = cv2.boundingRect(triangle_1)
    x,y,w,h = rect_1
    cropped_triangle_1 = frame[y: y+h, x: x+w]

    #------------create a mask--------
    mask_1 = np.zeros((h,w),np.uint8)

    
    
    points = np.array( [[tr1_pt1[0]-x, tr1_pt1[1]-y],
                        [tr1_pt2[0]-x, tr1_pt2[1]-y],
                        [tr1_pt3[0]-x, tr1_pt3[1]-y]], np.int32)
    cv2.fillConvexPoly(mask_1,points,255)

    #-------- apply the mask------
    cropped_triangle_1 = cv2.bitwise_and(cropped_triangle_1,cropped_triangle_1,mask = mask_1)

    # for 2nd img
    tr2_pt1 = landmarks_points_2[triangle_index[0]]
    tr2_pt2 = landmarks_points_2[triangle_index[1]]
    tr2_pt3 = landmarks_points_2[triangle_index[2]]
    triangle_2 = np.array([tr2_pt1,tr2_pt2,tr2_pt3], np.int32)
    # mask_2 = np.zeros_like(triangle_2)
    
    rect_2 = cv2.boundingRect(triangle_2)
    x2,y2,w2,h2 = rect_2
    cropped_triangle_2 = frame_2[y2: y2+h2, x2: x2+w2]

    #----------- create a mask--------
    mask_2 = np.zeros((h2,w2),np.uint8)

    # cv2.line(frame,tr2_pt1,tr2_pt2,(0,255,0),2)
    # cv2.line(frame,tr2_pt2,tr2_pt3,(0,255,0),2)
    # cv2.line(frame,tr2_pt1,tr2_pt3,(0,255,0),2)
    
    points_2 = np.array([[tr2_pt1[0]-x2, tr2_pt1[1]-y2],
                        [tr2_pt2[0]-x2, tr2_pt2[1]-y2],
                        [tr2_pt3[0]-x2, tr2_pt3[1]-y2]], np.int32)
    cv2.fillConvexPoly(mask_2,points_2,255)

    #-------- apply the mask------
    cropped_triangle_2 = cv2.bitwise_and(cropped_triangle_2,cropped_triangle_2,mask = mask_2)


    # cv2.line(frame_2,tr2_pt1,tr2_pt2,(0,255,0),2)
    # cv2.line(frame_2,tr2_pt2,tr2_pt3,(0,255,0),2)
    # cv2.line(frame_2,tr2_pt1,tr2_pt3,(0,255,0),2)

    #-------- warp triangles------
    points = np.float32(points)
    points_2 = np.float32(points_2)

    # fine transformation
    Matrix = cv2.getAffineTransform(points,points_2)
    # print(Matrix[0])
    warped_triangle = cv2.warpAffine(cropped_triangle_1,Matrix,(w2,h2))

    # reconstruct destination face
    triangle_area = frame_2_new_face[y2:y2+h2, x2:x2+w2]
    triangle_area = cv2.add(triangle_area, warped_triangle)

    frame_2_new_face[y2:y2+h2, x2:x2+w2] = triangle_area

#---- face swapping -------
# first remove the face from 2nd img and make it black
frame_2_new_face_gray = cv2.cvtColor(frame_2_new_face,cv2.COLOR_BGR2GRAY)
_,background = cv2.threshold(frame_2_new_face_gray,1,255,cv2.THRESH_BINARY_INV)


background = cv2.bitwise_and(frame_2,frame_2,mask = background)
result = cv2.add(background,frame_2_new_face)


cv2.imshow("frame",frame)
cv2.imshow("frame_2",frame_2)
# cv2.imshow("cropped_triangle_1",cropped_triangle_1)
# cv2.imshow("cropped_triangle_2",cropped_triangle_2)
# cv2.imshow("mask_1", mask_1)
# cv2.imshow("warped_triangle", warped_triangle)
# cv2.imshow("frame_2_new_face", frame_2_new_face)
# cv2.imshow("background", background)
cv2.imshow("result", result)


cv2.waitKey(0)

cv2.destroyAllWindows()