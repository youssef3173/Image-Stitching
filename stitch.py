
import numpy as np
import cv2
import random 


from util import extract_features, ransac, drawMatches, calculateHomography




##########################################################################################################
##########################################################################################################

path1 = 'Hill1.JPG'
path2 = 'Hill2.JPG'
path3 = 'Hill3.JPG'

img11 = cv2.imread( path1)  
img22 = cv2.imread( path2) 
img33 = cv2.imread( path3) 

img11 = cv2.cvtColor(img11, cv2.COLOR_BGR2RGB)
img22 = cv2.cvtColor(img22, cv2.COLOR_BGR2RGB)
img33 = cv2.cvtColor(img33, cv2.COLOR_BGR2RGB)

img1 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img33, cv2.COLOR_BGR2GRAY)

##########################################################################################################
# matches img1 -> img2 :

corrs, kp1, kp2, matches = extract_features( img1, img2) 
threshold = 0.50
finalH, inliers = ransac(corrs, threshold)
imgMatch, corresp12 = drawMatches( img1, kp1, img2, kp2, matches, inliers)


print( 'Number of correspendances between image 1 and image 2: {} \n '.format(len( corresp12)) )

out_img = imgMatch
cols1 = img1.shape[1]

for pair in corresp12 : 
  cv2.circle( out_img, pair[0], 4, ( 0, 0, 255), -1)
  cv2.circle( out_img, ( pair[1][0] + cols1, pair[1][1]), 4, ( 0, 0, 255), -1)

cv2.imwrite( 'MATCHES-1-2.JPG', out_img )


##########################################################################################################
# matches img3 -> img2 :

corrs, kp1, kp2, matches = extract_features( img3, img2) 
threshold = 0.50
finalH, inliers = ransac(corrs, threshold)
imgMatch, corresp32 = drawMatches( img3, kp1, img2, kp2, matches, inliers)


##########################################################################################################
# Calculate Homography for N points:

L = []
for item in corresp12:
    L.append( [ item[0][0], item[0][1], item[1][0], item[1][1]] )
C = np.matrix( L)
Hom12 = calculateHomography( C)

L = []
for item in corresp32:
    L.append( [ item[0][0], item[0][1], item[1][0], item[1][1]] )
C = np.matrix( L)
Hom32 = calculateHomography( C)


##########################################################################################################
# Stitching:

# Image Warping : 2 images so far,
# This algorithm depend on the relative position ofthe 2 images, (left and right)
# img2 is the main image : img1(warped) | img2 | img3(warped)

h_in_1 = img1.shape[0]
w_in_1 = img1.shape[1]

h_in_2 = img2.shape[0]
w_in_2 = img2.shape[1]

h_in_3 = img3.shape[0]
w_in_3 = img3.shape[1]

pts1 = [ [ 1, 1, 1], [ w_in_1, 1, 1], [ 1, h_in_1, 1], [ w_in_1, h_in_1, 1] ]
pts3 = [ [ 1, 1, 1], [ w_in_3, 1, 1], [ 1, h_in_3, 1], [ w_in_3, h_in_3, 1] ]

pts12 = pts1
for i in range( len(pts1)):
    pts12[i] = np.dot( Hom12, np.transpose(np.matrix( pts1[i])) )
    pts12[i] = pts12[i]/pts12[i][2][0]

pts32 = pts3
for i in range( len(pts3)):
    pts32[i] = np.dot( Hom32, np.transpose(np.matrix( pts3[i])) )
    pts32[i] = pts32[i]/pts32[i][2][0]

x_max = max( int(pts12[0][0][0]), int(pts12[1][0][0]), int(pts12[2][0][0]), int(pts12[3][0][0]), w_in_2 )
y_max = max( int(pts12[0][1][0]), int(pts12[1][1][0]), int(pts12[2][1][0]), int(pts12[3][1][0]), h_in_2 )

x_min = min( int(pts12[0][0][0]), int(pts12[1][0][0]), int(pts12[2][0][0]), int(pts12[3][0][0]), 0)
y_min = min( int(pts12[0][1][0]), int(pts12[1][1][0]), int(pts12[2][1][0]), int(pts12[3][1][0]), 0)

x_max = max( int(pts32[0][0][0]), int(pts32[1][0][0]), int(pts32[2][0][0]), int(pts32[3][0][0]), x_max)
y_max = max( int(pts32[0][1][0]), int(pts32[1][1][0]), int(pts32[2][1][0]), int(pts32[3][1][0]), y_max )

x_min = min( int(pts32[0][0][0]), int(pts32[1][0][0]), int(pts32[2][0][0]), int(pts32[3][0][0]), x_min)
y_min = min( int(pts32[0][1][0]), int(pts32[1][1][0]), int(pts32[2][1][0]), int(pts32[3][1][0]), y_min)

w_out = x_max - x_min + 1
h_out = y_max - y_min + 1 

res = np.zeros( [ h_out, w_out, 3], dtype= int)
print( 'out shape : {} '.format( res.shape) )

for x in range( x_min, x_max+1):
    for y in range( y_min, y_max+1):
        p1 = np.transpose( np.matrix([ x, y, 1]))
        p2 = np.dot( np.linalg.inv( Hom12), p1 )

        p =  p2/p2[2][0] 
        new_x_12 = int(p[0][0]) 
        new_y_12 = int(p[1][0]) 

        p2 = np.dot( np.linalg.inv( Hom32), p1 )

        p =  p2/p2[2][0] 
        new_x_32 = int(p[0][0]) 
        new_y_32 = int(p[1][0]) 

        for i in range(3):  # 3 -> RGB
            if ( 0 <= new_x_12 < w_in_1) and (0 <= new_y_12 < h_in_1 ) :
                res[y-y_min][x-x_min][i] = int( img11[new_y_12][new_x_12][i]) 
            elif ( 0 <= new_x_32 < w_in_3) and (0 <= new_y_32 < h_in_3 ):
                res[y-y_min][x-x_min][i] = int( img33[new_y_32][new_x_32][i] )
            elif ( 0 <= x < w_in_2) and (0 <= y < h_in_2 ):
                res[y-y_min][x-x_min][i] = int( img22[y][x][i] )
            else:
                res[y-y_min][x-x_min][i] = 0
                
                
                
##########################################################################################################
# Save results:

res = res.astype( np.uint8)
cv2.imwrite( 'res.JPG', res)

##########################################################################################################



