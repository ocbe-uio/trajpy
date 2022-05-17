import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
#import tamanho

def captura(file_name,number):
	cap = cv2.VideoCapture(0)
	#cap = cv2.VideoCapture('Projeto sem Título.mp4')
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	fps = cap.get(cv2.CAP_PROP_FPS)
	#############################################
	'''
	result, image = cap.read()
	if result:

		# showing result, it take frame name and image 
		# output
		cv2.imshow("Photo_sample", image)

		# saving image in local storage
		#cv2.imwrite("sample_photo.png", image)
		bbox = cv2.selectROI(image, False)
		# If keyboard interrupt occurs, destroy image 
		# window
		cv2.waitKey(1) #press enter to continue
		cv2.destroyWindow("Photo_sample")
		'''
#######################################################3

	#px_per_cm = tamanho.size(cap)/10
	#salvar = cv2.VideoWriter('teste_live.avi',cv2.VideoWriter_fourcc('P','I','M','1'), fps, (width, height), isColor=True)
	# Create the background subtractor object
	# Use the last 700 video frames to build the background
	back_sub = cv2.createBackgroundSubtractorKNN()
	#history=700,         varThreshold=25, detectShadows=False

	# Create kernel for morphological operation
	# You can tweak the dimensions of the kernel
	# e.g. instead of 20,20 you can try 30,30.
	kernel = np.ones((20,20),np.uint8)
	arquivo = open(str(file_name)+'.txt','w') #arquivo externo
	numero_frames = 0
	while(True):
		numero_frames += 1
		# Capture frame-by-frame
		# This method returns True/False as well
		# as the video frame.
		ret, frame = cap.read()
		if not ret:
			break

		# Use every frame to calculate the foreground mask and update
		# the background
		
		fg_mask = back_sub.apply(frame) #####

		# Close dark gaps in foreground object using closing
		fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel) #####

		# Remove salt and pepper noise with a median filter
		fg_mask = cv2.medianBlur(fg_mask, 5)  ####

		# Threshold the image to make it either black or white
		_, fg_mask = cv2.threshold(fg_mask,127,255,cv2.THRESH_BINARY) ####

		# Find the index of the largest contour and draw bounding box
		fg_mask_bb = fg_mask ###
		contours, hierarchy = cv2.findContours(fg_mask_bb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:] ####
		areas = [cv2.contourArea(c) for c in contours] 

		# If there are no countours
		if len(areas) < 1:

		# Display the resulting frame
			#cv2.imshow('frame',frame)

		# If "q" is pressed on the keyboard, 
		# exit this loop
			#if cv2.waitKey(1) & 0xFF == ord('q'):
			#    break

		# Go to the top of the while loop
			continue

		else:
		# Find the largest moving object in the image
			max_index = np.argmax(areas)

		# Draw the bounding box
		cnt = contours[max_index]
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

		#timer = cv2.getTickCount()
		#fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

		#px_per_cm = bbox[2]/float(number)
		px_per_cm = w/float(number)
		# Draw circle in the center of the bounding box
		x2 = x + int(w/2)
		y2 = y + int(h/2)
		if numero_frames%5 == 0:
			arquivo.write(str(np.round(numero_frames/fps,2))+','+str(np.round(x2/px_per_cm,2))+','
			+str(np.round(y2/px_per_cm,2))+'\n') 
		cv2.circle(frame,(x2,y2),4,(0,255,0),-1)

		# Print the centroid coordinates (we'll use the center of the
		# bounding box) on the image
		text = "x: " + str(np.round(x2/px_per_cm,2)) + ", y: " + str(np.round(y2/px_per_cm,2)) 
		cv2.putText(frame, text, (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0 ,255), 2)
		
		#w_text = 'Width:' + str(w)
		#cv2.putText(frame, w_text, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		
		
		fps_text = 'Tempo = ' + str(np.round(numero_frames/fps,2))
		cv2.putText(frame,fps_text,(120,80),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

		stop_text = 'Press "q" to stop tracking'
		cv2.putText(frame,stop_text,(120,120),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

		# Display the resulting frame
		#salvar.write(frame)
		cv2.imshow('frame',frame)
		#cv2.imwrite('frame'+str(numero_frames)+'.jpg',frame)
		

		# If "q" is pressed on the keyboard, 
		# exit this loop
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Close down the video stream
	cap.release()
	cv2.destroyAllWindows()
	arquivo.close()
	#salvar.release()
	
if __name__ == '__main__':
    captura()
