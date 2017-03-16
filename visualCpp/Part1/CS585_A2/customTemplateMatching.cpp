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

// Global Variables
Mat s_img;
Mat t_img;
Mat result;

const char* tempFileName = "pen2.jpg";
const char* image_window = "Source Image";
const char* result_window = "Result window";

// Template Matching Method
void tempMatchingNCC(Mat& s_d_Img, Mat& t_d_Img, Mat& ncc_result);

// main function
int main()
{
	// Mat object for video frame 0
	Mat Vframe0;

	// Mat object for grayscale images
	Mat t_g_dst;
	Mat s_g_dst;

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
	bool bSuccess0 = videoCh.read(Vframe0);

	//if not successful, break loop
	if (!bSuccess0)
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

	// Create windows
	namedWindow(image_window, WINDOW_AUTOSIZE);
	namedWindow(result_window, WINDOW_AUTOSIZE);

	// variable used inside the while loop
	bool bSuccess;


	// Convert template image to grayscale
	t_g_dst = Mat::zeros(t_img.rows, t_img.cols, CV_8UC1);
	cvtColor(t_img, t_g_dst, CV_BGR2GRAY);

	// Pyramid method
	//http://docs.opencv.org/doc/tutorials/imgproc/pyramids/pyramids.html

	// while no esc key is entered, loop
	while (1)
	{
		// Read a new image from window
		bSuccess = videoCh.read(s_img);

		//if video frame capture is not successful, break loop
		if (!bSuccess)
		{
			cout << "### Cannot read frame from video" << endl;
			break;
		}

		imshow(image_window, s_img);

		// Convert search image into grayscale s_g_dst
		s_g_dst = Mat::zeros(s_img.rows, s_img.cols, CV_8UC1);
		cvtColor(s_img, s_g_dst, CV_BGR2GRAY);

		// OpenCV Callback function for Template Matching function at each frame
		//tempMatchingOpenCV(0, 0);

		vector<Mat> s_downImg, t_downImg, results;

		int maxlevel = 3;

		// Build Gaussian pyramid
		//http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=buildpyramid#buildpyramid
		buildPyramid(s_g_dst, s_downImg, maxlevel);
		buildPyramid(t_g_dst, t_downImg, maxlevel);


		// testing
		imshow("Display window s", s_downImg[2]);
		imshow("Display window t", t_downImg[2]);

		Mat ref, tpl, res;

		// Process each pyramid
		//for (int level = maxlevel; level >= 0; level--)
		//{
			int idx = 3;
			ref = s_downImg[idx];
			tpl = t_downImg[idx];
			//res = Mat::zeros(ref.size() + Size(1, 1) - tpl.size(), CV_32FC1);

			Mat ncc_result;
			// Create the result matrix
			int result_cols = ref.cols - tpl.cols + 1;
			int result_rows = ref.rows - tpl.rows + 1;

			ncc_result.create(result_rows, result_cols, CV_32FC2);

			// Template Matching function with NCC
			tempMatchingNCC(ref, tpl, ncc_result);

		//}
		
		if (waitKey(30) == 27)
		{
			cout << "### Exit Program" << endl;
			break;
		}
	}

	videoCh.release();
	return 0;
}



/*-------------------------------------------------------------
Function: tempMathcingNCC()

@param

- This funciton performs template matching using NCC
-------------------------------------------------------------*/
void tempMatchingNCC(Mat& s_d_Img, Mat& t_d_Img, Mat& ncc_result)
{
	int i, j, x, y;			// variable for for-loops

	// variables used in the for loop
	int n = 0;					// number of pixels in a template image
	double NCC_r = -10;			// r value calculated 
	double minNCC = 0;			// minimum r_value
	double sumTemp = 0;			// temporary storage for calculating NCC
	int s_intensity;	
	int t_intensity;

	// variable for calculating NCC
	Scalar s_mean, s_stddev;	// mean and standard deviation of search image
	Scalar t_mean, t_stddev;	// mean and standard deviation of template image

	// variable for storing top matchinging location
	int bestRow = 0;			// row with highest NCC value
	int bestCol = 0;			// col with highest NCC value
	double bestNCC = 0;			// best NCC value

//	Mat img_display; 			// for displaying image

//	s_img.copyTo(img_display);
		
	// Get number of pixels in a template image
	// openCV reference: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat-total
	n = t_d_Img.total();

	// Calculate mean and standard deviation of template image
	// http://docs.opencv.org/modules/gpu/doc/matrix_reductions.html#gpu-meanstddev
	meanStdDev(t_d_Img, t_mean, t_stddev, cv::Mat());
	
	// Compute NCC 'r' of the pixel location using following NCC equation:
	//
	//	r = 1/n * sum of ij [(S_ij - mean (S)) (T_ij - mean(T)) / (sd_s * sd_m)]
	//
	// where:
	//	template image 		=	T
	//	search image 		=	S 
	//	standard deviation	= 	sd
	//	s_i		= brightness of the ith pixel in search image
	//	t_i		= brightness of the ith pixel in template image
	//	mean(s) = mean of all pixels in SUB image
	//	mean(t) = mean of all pixels in template image
	//	sd_s	= standard deviation of all pixels in subimage
	//	sd_t	= standard deviation of all pixels in template image
	//	n		= number of pixels in template image 					
	
	cout << "### StartLoop" << endl;

	// Loop through search image
	for (x = 0; x < (s_d_Img.cols - t_d_Img.cols); x++)	{
		for (y = 0; y < (s_d_Img.rows - t_d_Img.rows); y++) {

			// Take a sub-image of the image
			Mat sub_img = s_d_Img(Rect(x, y, t_d_Img.cols, t_d_Img.rows));

			// Calculate mean and standard deviation of search image
			// http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#meanstddev
			meanStdDev(sub_img, s_mean, s_stddev, cv::Mat());

			sumTemp = 0;

			//loop through template image and calculate NCC
			for (j = 0; j < t_d_Img.cols; j++) {
				for (i = 0; i < t_d_Img.rows; i++) {

					s_intensity = s_d_Img.at<uchar>(x, y);
					t_intensity = t_d_Img.at<uchar>(i, j);

					// sum of ij [(S_ij - mean (S)) (T_ij - mean(T)) / (sd_s * sd_m)]		
					sumTemp += (s_intensity - s_mean.val[0]) * (t_intensity - t_mean.val[0]) / (s_stddev.val[0] * t_stddev.val[0]);
				}
			}

			NCC_r = sumTemp / n;

			ncc_result.at<Vec2d>(i,j) = NCC_r;

			// store the best template matching position
			if (minNCC < NCC_r)
			{
				minNCC = NCC_r;
				bestRow = x;
				bestCol = y;
				bestNCC = NCC_r;
			}

			cout << "### NCC_r" << NCC_r << " N: " << n << "### bestNCC " << bestNCC << endl;

		}
	}
	
	//cout << "### best x " << bestRow << " best y " << bestCol << endl;


	cout << "### End Loop" << endl;

	// Show me what you got
//	rectangle(img_display, Point(bestRow, bestCol), Point(bestRow + t_Img.cols, bestCol + t_Img.rows), Scalar::all(0), 2, 8, 0);
//	rectangle(result, Point(bestRow, bestCol), Point(bestRow + t_Img.cols, bestCol + t_Img.rows), Scalar::all(0), 2, 8, 0);

//	imshow(image_window, img_display);
//	imshow(result_window, result);

}

