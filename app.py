import streamlit as st
# import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from io import StringIO 

def main():
    st.image("./img.jpg",use_column_width=True)


    st.title("Crack Detection")
    st.subheader("Please upload image : ")
    image_filess=st.file_uploader("Upload Image",
                                type=["png","jpg","jpeg"], accept_multiple_files=True)

    # uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    
    if st.button("Process"):
        for image_file in image_filess:
            # image1=load_image(image_file)
            image1=Image.open(image_file)
            st.image(image1,caption="Uploaded Image")
            image=np.array(image1)
            
            sigma = 0.33
            median = np.median(image)
            lower = int(max(0, (1.0 - sigma) * median))
            upper = int(min(255, (1.0 + sigma) * median))
            #lower sigma-->tighter threshold(default value of sigma can be 0.33)4

            edge_image= cv2.Canny(image, lower, upper)
            # cv2_imshow(edge_image)

            # Canny Edge Detection
            edges = cv2.Canny(edge_image,100,200)

            # Morphological Closing Operator
            kernel = np.ones((5,5),np.uint8)
            closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Create feature detecting method
            orb = cv2.ORB_create(nfeatures=1500)

            # Make featured Image
            keypoints, descriptors = orb.detectAndCompute(closing, None)
            featuredImg = cv2.drawKeypoints(closing, keypoints, None)

            # Create an output image
            st.image(featuredImg,caption="Output")
            #print(img)
            # st.subheader(prediction_cls(model.predict(img)))

    pass

if __name__=="__main__":
    main()