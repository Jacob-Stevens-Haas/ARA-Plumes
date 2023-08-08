import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_max_on_boundary(array, center, r, rtol = 1e-3, atol=1e-6):
    col, row = center
    n, d = array.shape

    # Generate a grid of indices for the array
    xx, yy = np.meshgrid(np.arange(d), np.arange(n)) 

    # Calculate the Euclidean distance from each point to the center
    # Create array of distances from center 
    distances = np.sqrt((xx - col) ** 2 + (yy - row) ** 2)

    # Create a mask for points on the boundary (distances == r)
    boundary_mask = np.isclose(distances, r, rtol=rtol,atol=atol)
    # boundary_mask = np.isclose(distances,r)

    # Apply the boundary mask to the array to get the subset
    boundary_subset = array[boundary_mask]

    # Find the maximum value within the subset
    max_value = np.max(boundary_subset)

    # Find the indices of the maximum elements within the boundary
    max_indices = np.argwhere(np.isclose(array, max_value) & boundary_mask)

    row, col = max_indices[0]

    max_indices = (col, row)


    # Return the maximum value and its locations
    return max_value, max_indices


def find_next_center(array, orig_center, neig_center, r, scale=3/4, rtol=1e-3, atol=1e-6):
    col, row = orig_center
    n, d = array.shape

    # print("curr center:", (col,row))

    # generate grid of indicies from the array
    xx, yy = np.meshgrid(np.arange(d), np.arange(n))

    # Get array of distances 
    distances = np.sqrt((xx-col)**2 + (yy-row)**2)

    # Create a mask for points on the boundary (distances == r)
    boundary_mask = np.isclose(distances,r,rtol=rtol,atol=atol)
    # boundary_mask = np.isclose(distances,r)


    # create interion on previous circle
    col, row = neig_center

    # print("prev center:", (col,row))

    # get array of new distnaces from previous circle
    distances = np.sqrt((xx-col)**2 + (yy-row)**2)

    # Create interior mask
    interior_mask = distances <= r*scale

    search_mask = boundary_mask & interior_mask

    search_subset = array[search_mask]

    print("Search subset:",search_subset)

    max_value = np.max(search_subset)

    # find indicies of max element
    max_indicies = np.argwhere(np.isclose(array,max_value) & search_mask)

    row, col = max_indicies[0]

    max_indicies = (col,row)

    return max_value, max_indicies


def main():

    # include rings in plot?
    boundary_ring = False
    interior_ring = False
    fit_poly = True

    # isclose parameters
    # somthign about rtol=1e-2 gives just zeros -- figure out why but not 1e-1 or 1e-3
    rtol = 1e-1
    atol = 1e-5
    print("rtol:", rtol)

    # Read image in 
    img_path = "/Users/Malachite/Documents/UW/ARA/Plumes/video_1242/fixed_avg_frames/subtract_10465.png"

    # problematic frame
    img_path = "/Users/Malachite/Documents/UW/ARA/Plumes/video_1242/fixed_avg_frames/subtract_10598.png"
    img_path = "/Users/Malachite/Documents/UW/ARA/Plumes/July_20/video_low_1/fixed_avg_frames/subtract_0383.png"
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Pick and draw initial center (in blue)
    center = (170, 135) # column, row
    center = (1590,1000)
    cir_radius = 3
    blue_color = (255,0,0)
    thickness = -1

    cv2.circle(img,center, cir_radius, blue_color, thickness)


    # Select and plot first circle 
    num_of_circs = 25
    radii = 50

    points = np.zeros(shape=(num_of_circs+1,2))
    points[0]=center
    # Find first center

    orig_center = center 
    _, center = find_max_on_boundary(array=gray, center=center, r=radii, rtol=rtol,atol=atol)
    points[1]=center

    # Plot red circle
    cir_radius = 3
    red_color = (0,0,255)
    thickness = -1
    cv2.circle(img=img, center= center, radius=cir_radius, color=red_color, thickness=thickness)

    # Draw the ring
    if boundary_ring == True:
        cv2.circle(img=img, center=orig_center, radius=radii, color=red_color, thickness=1, lineType=cv2.LINE_AA)


    scale = 3/5

    for step in range(2,num_of_circs+1):
        radius = radii*step

        # draw interior ring
        if interior_ring == True:
            cv2.circle(img,
                       center = center,
                       radius = int(radius*scale),
                       color = blue_color,
                       thickness = 1,
                       lineType = cv2.LINE_AA)

        # Get center of next point
        _, center = find_next_center(array=gray,
                                             orig_center=orig_center,
                                             neig_center=center,
                                             r=radius,
                                             scale=scale,
                                             rtol=rtol,
                                             atol=atol)
        
        points[step]=center

        # Draw new red point
        cv2.circle(img,center,cir_radius,red_color,thickness)

        # draw boundary ring
        if boundary_ring == True:
            cv2.circle(img,center = orig_center, radius = radius, color=red_color,thickness=1,lineType=cv2.LINE_AA)
    
    
    # print(points)
    poly_coef = np.polyfit(points[:,0],points[:,1],deg=2)
    # print(poly_coef)
    
    # plot points and polynomial

    # generate x values for plotting the polynomial curve
    x = np.linspace(np.min(points[:,0]),np.max(points[:,0])+70,100)

    # compute y values for learned polyomial curve
    y = poly_coef[0]*x**2 + poly_coef[1]*x + poly_coef[2]

    curve_img = np.zeros_like(img)

    curve_points = np.column_stack((x,y)).astype(np.int32)

    # draw poly curve on image
    cv2.polylines(curve_img,[curve_points], isClosed=False,color=(255,0,0), thickness=2)

    if fit_poly == True:
        img = cv2.addWeighted(img,1,curve_img,1,0)

    # plot results 
    # plt.scatter(points[:,0], points[:,1], label="Points")
    # plt.plot(x,y, color='red', label="polynomial")
    # plt.xlabel('x')
    # plt.ylabel('y')

    # # Flip the y-axis
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.show()
    cv2.imshow("focused concentric search", img)


        
    # #############3
    # img = cv2.imread(img_path)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # add center in as red dot
    # center = (170, 135)
    # cir_radius = 3
    # red_color = (255, 0, 0)
    # thickness = -1

    # cv2.circle(img, center, cir_radius, red_color, thickness)

    # for step in range(num_of_circs):
    #     step += 1
    #     radius = radii * step

    #     # Get center of next image
    #     max_value, new_center = find_max_on_boundary(array=gray, center=center, r=radius)

    #     # # Take note we had to switch the y and x
    #     # y, x = new_center

    #     # # add blue dot to image
    #     # print(f"{new_center}: {max_value}")
    #     cir_radius = 3
    #     blue_color = (0, 0, 255)
    #     thickness = -1
    #     cv2.circle(img=img, center=new_center, radius=cir_radius, color=blue_color, thickness=thickness)

    #     # Draw boundary ring
    #     if boundary_ring == True:
    #         cv2.circle(img=img, center=center, radius=radius, color=blue_color, thickness=1, lineType=cv2.LINE_AA)

    #     cv2.imshow("concentric search", img)

    # ################
    # img = cv2.imread(img_path)
    # cv2.imshow("orig", img)

    # ################


    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()







# def main():
#     img_path = "/Users/Malachite/Documents/UW/ARA/Plumes/video_1242/fixed_avg_frames/subtract_10365.png"
#     img = cv2.imread(img_path)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # add center in as red dot
#     center = (170, 135)
#     cir_radius = 3
#     red_color = (255, 0, 0)
#     thickness = -1

#     cv2.circle(img, center, cir_radius, red_color, thickness)

#     num_of_circs = 6
#     radii = 35

#     for step in range(num_of_circs):
#         step += 1
#         radius = radii * step

#         # Get center of next image
#         max_value, new_center = find_max_on_boundary(array=gray, center=center, r=radius)

#         # # Take note we had to switch the y and x
#         # y, x = new_center

#         # # add blue dot to image
#         # new_center = (x, y)
#         # print(new_center)
#         print(f"{new_center}: {max_value}")
#         cir_radius = 3
#         blue_color = (0, 0, 255)
#         thickness = -1
#         cv2.circle(img=img, center=new_center, radius=cir_radius, color=blue_color, thickness=thickness)

#         # Draw the ring
#         cv2.circle(img=img, center=center, radius=radius, color=blue_color, thickness=1, lineType=cv2.LINE_AA)

#     # save image
#     # output_path = "center_path.png"
#     # cv2.imwrite(output_path, img)

#     cv2.imshow("centers", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# import cv2
# import numpy as np


# def find_max_on_boundary(array, center, r):
#     x,y=center
#     n, d = array.shape

#     # Generate a grid of indices for the array
#     xx, yy = np.meshgrid(np.arange(d), np.arange(n))

#     # Calculate the Euclidean distance from each point to the center
#     distances = np.sqrt((xx - x)**2 + (yy - y)**2)

#     # Create a mask for points on the boundary (distances == r)
#     boundary_mask = np.isclose(distances, r)

#     # Apply the boundary mask to the array to get the subset
#     boundary_subset = array[boundary_mask]

#     # Find the maximum value within the subset
#     max_value = np.max(boundary_subset)

#     # Find the indices of the maximum elements within the boundary
#     max_indices = np.argwhere(np.isclose(array, max_value) & boundary_mask)

#     # Return the maximum value and its locations
#     return max_value, max_indices


# def main():

#     img_path = "/Users/Malachite/Documents/UW/ARA/Plumes/video_1242/fixed_avg_frames/subtract_10465.png"
#     img = cv2.imread(img_path)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # add center in as red dot
#     center = (170,135)
#     cir_radius = 3
#     red_color = (255,0,0)
#     thickness = -1

#     cv2.circle(img,
#             center,
#             cir_radius,
#             red_color,
#             thickness)


#     num_of_circs = 10
#     radii = 13

#     for step in range(num_of_circs):
#         step += 1
#         radius = radii*step

#         # Get center of next image
#         _, new_center = find_max_on_boundary(array=gray,
#                                             center=center,
#                                             r = radius)
        
#         # Take note we had to switch the y and x
#         y,x = new_center[0]

#         # add blue dot to image
#         new_center = (x,y)
#         print(center)
#         cir_radius = 3
#         blue_color = (0,0,255)
#         thickness = -1
#         cv2.circle(img=img,
#                 center=new_center,
#                 radius=cir_radius,
#                 color=blue_color,
#                 thickness=thickness)
        
#     # save image
#     output_path = "center_path.png"

#     cv2.imwrite(output_path,img)
        
#     cv2.imshow("centers", img)

#     cv2.waitKey(0)

#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()


#######################################################################################################################################