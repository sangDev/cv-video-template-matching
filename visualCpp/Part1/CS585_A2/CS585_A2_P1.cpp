/*----------------------------------------------------------------
	CS585_A2_main.cpp

	CS585 Image and Video Computing Fall 2015
	Assignment : A2

	PART 1: Object Detection with Template Matching

	This assignment has been broken down into following subparts.

	PART A		:	Open video channel and stream images from webcamera
	PART B		:	Load template image of an object and perform template matching
	Part C		:	Indicate graphically where the object is and draw bounding box

	Author:	Sang-Joon Lee
	Date:	Sept 22, 2015

----------------------------------------------------------------- */

#include "stdafx.h"

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

//C++ standard libraries
#include <iostream>
#include <vector>

// Namespace
using namespace cv;
using namespace std;

// File names & Window Names
const char* tempFileName = "pen3.jpg";
const char* image_window = "Source Image";
const char* result_window = "Result window";

// Template Matching Method
void tempMatchingOpenCV(Mat& s_img, Mat& t_img);

// main function
int main()
{
	// variable used to video capture
	bool captureSucess;
	bool open0;

	// variable for search and template image
	Mat s_img;
	Mat t_img;

	// Mat object for video frame 0
	Mat Vframe0;
		
	// -------------------------------------------------------------------
	// A.1 Open Video Channel to read stream of images from a webcamera
	// --------------------------------------------------------------------
	// For more information on reading and writing video: 
	// http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html

	// Open Video channel
	VideoCapture videoCh(0);

	// if video open is not successful, exit program
	if (!videoCh.isOpened())
	{
		cout << "### Cannot open the video camera - check video installtion/driver" << endl;
		return -1;
	}

	// -------------------------------------------------------------------	
	// A.2 read a new frame from video
	// -------------------------------------------------------------------	
	open0 = videoCh.read(Vframe0);

	//if not successful, break loop
	if (!open0)
	{
		cout << "Cannot read a frame from video stream" << endl;
	}

	//create a window for video frame 0 and show the frame on the window
	namedWindow("VideoFrame0", WINDOW_AUTOSIZE);
	imshow("VideoFrame0", Vframe0);

	// -------------------------------------------------------------------	
	// B. Template Matching using Normalized Cross-Correlation
	// -------------------------------------------------------------------	

	// read template image file
	t_img = imread(tempFileName, IMREAD_COLOR);

	//create a window for template file
	namedWindow("template", WINDOW_AUTOSIZE);
	imshow("template", t_img);

	// Create windows for image and result
	namedWindow(image_window, WINDOW_AUTOSIZE);
	namedWindow(result_window, WINDOW_AUTOSIZE);
	
	// video capture and template matching loop
	while (1)
	{
		// read new image from video camera
		captureSucess = videoCh.read(s_img);

		// video capture failed
		if (!captureSucess)
		{
			cout << "### Cannot read frame from video" << endl;
			break;
		}		

		/// Call Template Matching function at each frame
		// pass the template image and search image from the video
		tempMatchingOpenCV(s_img, t_img);
		
		if (waitKey(30) == 27)
		{
			cout << "### Exit Program" << endl;
			break;
		}
	}

	videoCh.release();
	return 0;
}


/*------------------------------------------------
Function: tempMatchingOpenCV
	This function performs template matching using the OpenCV template matching function
	
	@param - s_img - search image 
	@param - t_img - template image 

	// Improvement need to be made on template matching to implement with
	pyramid building - sample down and detect
	http://docs.opencv.org/doc/tutorials/imgproc/pyramids/pyramids.html
-------------------------------------------------*/
void tempMatchingOpenCV(Mat& s_img, Mat& t_img)
{

	int match_method = 5;	//	CV_TM_CCOEFF_NORMED
	
	double minValue;		// 	Min NCC value
	double maxValue;		// 	Max NCC value
	Point minLocation;		// 	Min NCC location
	Point maxLocation;		// 	Max NCC location
	Point matchLocation;	// 	Matching object location

	int result_cols;		//	number of cols for resulting NCC 	
	int result_rows;		// 	number of rows for resulting NCC

	Mat imgDisplay;			//	container for displaying image
	Mat result;				//	container for normalized correlation coefficient 
	
	
	// Copy the original image
	s_img.copyTo(imgDisplay);

	// Create the result matrix
	result_cols = s_img.cols - t_img.cols + 1;
	result_rows = s_img.rows - t_img.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);

	// Perform template matching
	// openCV reference: http://docs.opencv.org/modules/imgproc/doc/object_detection.html#matchtemplate
	matchTemplate(s_img, t_img, result, match_method);

	//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	// get the best match by finding the max
	// http://docs.opencv.org/modules/core/doc/operations_on_arrays.html?highlight=minmaxloc#minmaxloc
	minMaxLoc(result, &minValue, &maxValue, &minLocation, &maxLocation, Mat());
	
	// Maximum NCC = match location
	matchLocation = maxLocation;
	cout << "MAX NCC VALUE: " << maxValue << endl;
		
	// Draw Rectangle on the Live Image Display
	rectangle(imgDisplay, matchLocation, Point(matchLocation.x + t_img.cols, matchLocation.y + t_img.rows), Scalar::all(0), 2, 8, 0);
	// Draw Rectangle on the resulting image display
	rectangle(result, matchLocation, Point(matchLocation.x + t_img.cols, matchLocation.y + t_img.rows), Scalar::all(0), 2, 8, 0);

	// Display Live Image with object detection
	imshow(image_window, imgDisplay);
	// Display Live result image with object detection
	imshow(result_window, result);

}
