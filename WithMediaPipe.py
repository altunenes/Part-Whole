# # Note: Your images need to be in the same dimensions, I couldn't have solved this issue in my first attempt. 

import mediapipe as mp
import cv2
import numpy as np
import streamlit as st
def index_np(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index
mp_face=mp.solutions.face_mesh
mesh=mp_face.FaceMesh()

#OPENCV image bgr default mediapipe rgb

img=cv2.imread("train/testface.jpg")
img2 = cv2.imread("train/otherface.jpg")

if img.shape!=img2.shape:
    print("img1 shape: ",img.shape)
    print("img2 shape:",img2.shape)
    raise ValueError("images must be same shape!")


mask = np.full_like(img,255)

height,weight,channel=img.shape

img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gaus=cv2.GaussianBlur(img_rgb,(7,7),60)

result=mesh.process(img_rgb)
indexes_triangles = []

# img2=cv2.resize(img2,(500,500))
uz,gen,kan=img2.shape
img_rgb2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img2_gray=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
result2=mesh.process(img_rgb2)

img2_new_face = np.zeros((height, weight, channel), np.uint8)


for i in result.multi_face_landmarks:
    landmarks_points = []

    # print(len(i.landmark))
    for j in range(0,468):
        pt1=i.landmark[j]
        x=int(pt1.x*weight)
        y=int(pt1.y*height)
        landmarks_points.append((x, y))
        # cv2.circle(img,(x,y),1,(0,255,0),1)
        # cv2.putText(img,str(j),(x,y),0,1,(0,0,0))
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask, convexhull, 255)
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = index_np(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = index_np(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = index_np(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)


for i in result2.multi_face_landmarks:
    landmarks2_points = []

    # print(len(i.landmark))
    for j in range(0,468):
        pt1=i.landmark[j]
        x=int(pt1.x*gen)
        y=int(pt1.y*uz)
        landmarks2_points.append((x, y))
        # cv2.circle(img,(x,y),1,(0,255,0),1)
        # cv2.putText(img,str(j),(x,y),0,1,(0,0,0))
        points2 = np.array(landmarks2_points, np.int32)
        convexhull2 = cv2.convexHull(points2)

lines_space_mask = np.zeros_like(img_gaus)
lines_space_new_face = np.zeros_like(img2)
# Triangulation
for triangle_index in indexes_triangles:
    # Triangulation of the target face
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    cropped_triangle = img[y: y + h, x: x + w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)


    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

    # Lines
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
    cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)

    # Triangulation of face 2
    tr2_pt1 = landmarks2_points[triangle_index[0]]
    tr2_pt2 = landmarks2_points[triangle_index[1]]
    tr2_pt3 = landmarks2_points[triangle_index[2]]
    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


    rect2 = cv2.boundingRect(triangle2)
    (x, y, w, h) = rect2

    cropped_tr2_mask = np.zeros((h, w), np.uint8)

    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

    # warping proccess
    points = np.float32(points)
    points2 = np.float32(points2)
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)


    img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
    img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area



# swap proccessing (putting 1st(test) face into 2nd(target) face)
img2_face_mask = np.zeros_like(img2_gray)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)


img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)

result = cv2.add(img2_head_noface, img2_new_face)

(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))


##seamless processing for smooth colors. it makes changes more realistic

seamless = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
gauss=cv2.GaussianBlur(seamless,(5,5),5)
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.imwrite("train/result.jpg",gauss)

cv2.imshow("seamlessclone", gauss)
cv2.waitKey(0)


cv2.destroyAllWindows()
