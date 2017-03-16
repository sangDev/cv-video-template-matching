/*----------------------------------------------------------------
	CS585_A2_Part2.cpp

	CS585 Image and Video Computing Fall 2015
	Assignment : A2

	PART 2: Gesture Recognition

	This project takes a live video image from camera and performs
	gesture recognition against a number of template images

	Author:	Sang-Joon Lee
	Date:	Sept 22, 2015

----------------------------------------------------------------- */

// window library
//#include "stdafx.h"

//openCV libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//C++ standard libraries
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

//Global variables
int thresh = 128;
int max_thresh = 255;

const char* image_window = "Source Image";
const char* result_window = "Result window";

// Template Matching Method
void tempMatchingOpenCV(Mat& s_img, Mat& t_img, Mat& result);


// Function that returns the maximum of 3 integers
int myMax(int a, int b, int c);

// Function that returns the minimum of 3 integers
int myMin(int a, int b, int c);

// Function that detects whether a pixel belongs to the skin based on RGB values
void SkinDetect(Mat& src, Mat& dst);

//Function that does frame differencing between the current frame and the previous frame
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);

// Function that accumulates the frame differences for a certain number of pairs of frames
void myMotionEnergy(vector<Mat> mh, Mat& dst);

// main function
int main()
{
    // variable to hold template images from file
    Mat tmpl_thumbsup_img;
    Mat tmpl_thumbsdown_img;
    Mat tmpl_idxOne_img;
    Mat tmpl_idxTwo_img;
    Mat tmpl_idxThree_img;
    Mat tmpl_palm_img;
    Mat tmpl_fist_img;

    Mat s_img;
    Mat t_img;

    Mat results_img;
    Mat img_display;

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
    bool bSuccess0 = videoCh.read(Vframe0);

    //if not successful, break loop
    if (!bSuccess0)
    {
        cout << "Cannot read a frame from video stream" << endl;
    }

    //create a window for video frame 0 and show the frame on the window
    namedWindow("VideoFrame0", WINDOW_AUTOSIZE);
    imshow("VideoFrame0", Vframe0);

    // Read template images
    tmpl_thumbsup_img = imread("thumbsup.jpg", IMREAD_COLOR);
    tmpl_thumbsdown_img = imread("thumbsdown.jpg", IMREAD_COLOR);
    tmpl_idxOne_img = imread("index_finger.jpg", IMREAD_COLOR);
    tmpl_idxTwo_img = imread("two.jpg", IMREAD_COLOR);
    tmpl_idxThree_img = imread("three.jpg", IMREAD_COLOR);
    tmpl_palm_img = imread("palm.jpg", IMREAD_COLOR);
    tmpl_fist_img = imread("fist.jpg", IMREAD_COLOR);

    Mat walkImg;
    Mat stopImg;
    Mat runImg;
    Mat tmpImg1;
    Mat tmpImg2;
    tmpImg1 = imread("walk.jpg", IMREAD_COLOR);
    tmpImg2 = imread("stop.jpg", IMREAD_COLOR);
    resize(tmpImg1, walkImg, runImg.size(), 0.5, 0.5, CV_INTER_AREA);

    runImg = imread("run.png", IMREAD_COLOR);
    resize(tmpImg2, stopImg, runImg.size(), 0.5, 0.5, CV_INTER_AREA);


    //create a windows displaying image
    namedWindow("thumbsup", WINDOW_AUTOSIZE);
    namedWindow("thumbsdown", WINDOW_AUTOSIZE);
    namedWindow("index_finger", WINDOW_AUTOSIZE);
    namedWindow("Skin", WINDOW_AUTOSIZE);
    namedWindow("Contours", CV_WINDOW_AUTOSIZE);
    namedWindow("SubImage", CV_WINDOW_AUTOSIZE);
    namedWindow("SubImageResized", CV_WINDOW_AUTOSIZE);

    namedWindow("GUI", WINDOW_AUTOSIZE);


    imshow("thumbsup", tmpl_thumbsup_img);
    imshow("thumbsdown", tmpl_thumbsdown_img);
    imshow("index_finger", tmpl_idxOne_img);

    // Create windows
    namedWindow(image_window, WINDOW_AUTOSIZE);
    namedWindow(result_window, WINDOW_AUTOSIZE);

    // variable used inside the while loop
    bool bSuccess;
    Point matchLoc;

    // used for classification
    double maxVal_tup = 0;
    double maxVal_tdown = 0;
    double maxVal_one = 0;
    double maxVal_two = 0;
    double maxVal_three = 0;
    double maxVal_palm = 0;
    double maxVal_fist = 0;

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    // while no esc key is entered, loop
    while (1)
    {
        // Read a new image from window
        bSuccess = videoCh.read(s_img);

        s_img.copyTo(img_display);

        //if video frame capture is not successful, break loop
        if (!bSuccess)
        {
            cout << "### Cannot read frame from video" << endl;
            break;
        }

        // perform skin detection on the search image
        // destination frame
        Mat frameDest;
        frameDest = Mat::zeros(s_img.rows, s_img.cols, CV_8UC1); //Returns a zero array of same size as src mat, and of type CV_8UC1

        //---------------------------------------
        // B. Skin color detection
        //---------------------------------------
        SkinDetect(s_img, frameDest);
        imshow("Skin", frameDest);

        //---------------------------------------
        // C. Find contour and find the subimage to filter out of search image
        //---------------------------------------
        // Documentation for finding contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
        findContours(frameDest, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        Mat contour_output = Mat::zeros(frameDest.size(), CV_8UC3);

        // Find largest contour
        int maxsize = 0;
        int maxind = 0;
        Rect boundrec;

        for (int i = 0; i < contours.size(); i++)
        {
            // Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
            double area = contourArea(contours[i]);

            if (area > maxsize) {
                maxsize = area;
                maxind = i;
                boundrec = boundingRect(contours[i]);
            }
        }

        // Draw contours
        // Documentation for drawing contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
        drawContours(contour_output, contours, maxind, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
        drawContours(contour_output, contours, maxind, Scalar(0, 0, 255), 2, 8, hierarchy);

        /*	Size tempSize = boundrec.size();
            cout << "MAT Size"<< s_img.size()<<endl;

            cout << "Current Rec:"<< boundrec.size()<<endl;


            Size deltaSize( tempSize.width * 0.2f, tempSize.height * 0.2f );
            Point offset( deltaSize.width/2, deltaSize.height/2);
            boundrec += deltaSize;
            boundrec -= offset;

            cout << "New Rec"<< boundrec.size()<<endl;*/

        // Documentation for drawing rectangle: http://docs.opencv.org/modules/core/doc/drawing_functions.html
        rectangle(contour_output, boundrec, Scalar(0, 255, 0), 1, 8, 0);

        // Show in a window
        imshow("Contours", contour_output);

        //http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=boundingrect#boundingrect
        Mat subImage;

        //Take subImage of the image subImage(output, cv::Rect(x1, y1, x2-x1, y2-y1));
        subImage = s_img(boundrec);

        // Show in a window
        imshow("SubImage", subImage);
        //cout << "Got SubImage"<<endl;
        //cout << "Subimage Size"<< subImage.size()<<endl;
        //cout << "Subimage AREA "<< subImage.size().area()<<endl;



        double minValue;
        Point minLoc;
        Point maxLoc;

        // temp image holder
        Mat t_img2;
        Mat results_img1;

        //-------------------------
        // D. Classification
        //-------------------------
        // Resize the picture into a smaller size
        // http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#resize
//		resize(tmpl_thumbsup_img, t_img2, subImage.size(), 0.5, 0.5, CV_INTER_AREA);

        // Call Template Matching function at each frame
        // thumbs up image
//		cout << "tmpl_thumbsup_img Size"<< tmpl_thumbsup_img.size()<<endl;
//		cout << "tmpl_thumbsup_img AREA "<< tmpl_thumbsup_img.size().area()<<endl;


        /*	Mat tmmPyOri = tmpl_thumbsup_img;
            Mat tmpPyDst;
            While (tmpl_thumbsup_img.size().area() > subImage.size().area())
            {
                pyrDown( tmmPyOri, tmpPyDst, Size( tmmPyOri.cols/2, tmmPyOri.rows/2 )
            }
    */

//		tempMatchingOpenCV(subImage, t_img2, results_img);
//		cout << "Got SubImage"<<endl;

//		minMaxLoc(results_img, &minValue, &maxVal_tup, &minLoc, &maxLoc, Mat());

        // tmpl one
        resize(tmpl_idxOne_img, t_img2, subImage.size(), 0.5, 0.5, CV_INTER_AREA);
        tempMatchingOpenCV(subImage, t_img2, results_img);
        minMaxLoc(results_img, &minValue, &maxVal_one, &minLoc, &maxLoc, Mat());

        // tmpl two
//		resize(tmpl_idxTwo_img, t_img2, subImage.size(), 0.5, 0.5, CV_INTER_AREA);
        //tempMatchingOpenCV(subImage, t_img2, results_img);
        //minMaxLoc(results_img, &minValue, &maxVal_two, &minLoc, &maxLoc, Mat());

        // tmpl three
//		resize(tmpl_idxThree_img, t_img2, subImage.size(), 0.5, 0.5, CV_INTER_AREA);
        //tempMatchingOpenCV(subImage, t_img2, results_img);
        //minMaxLoc(results_img, &minValue, &maxVal_three, &minLoc, &maxLoc, Mat());

        // tmpl tmpl_palm_img
        resize(tmpl_palm_img, t_img2, subImage.size(), 0.5, 0.5, CV_INTER_AREA);
        tempMatchingOpenCV(subImage, t_img2, results_img);
        minMaxLoc(results_img, &minValue, &maxVal_palm, &minLoc, &maxLoc, Mat());

        // tmpl tmpl_fist_img
        resize(tmpl_fist_img, t_img2, subImage.size(), 0.5, 0.5, CV_INTER_AREA);
        tempMatchingOpenCV(subImage, t_img2, results_img);
        minMaxLoc(results_img, &minValue, &maxVal_fist, &minLoc, &maxLoc, Mat());


        matchLoc = maxLoc;

        //@ debug print out
        //cout << "UP: " << maxVal_tup << " 2:" << maxVal_two << " 1:" << maxVal_one << " #3: " << maxVal_three << " P: " << maxVal_palm << " F: " << maxVal_fist << endl;
        //double classfi[] = { maxVal_tup, maxVal_one, maxVal_two, maxVal_three, maxVal_palm, maxVal_fist };

        cout << " 1:" << maxVal_one << " P: " << maxVal_palm << " F: " << maxVal_fist << endl;
        double classfi[] = { maxVal_one, maxVal_palm, maxVal_fist };

        double max = maxVal_one;
        int idx = 0;

        for (int i = 0; i< 3; i++)
        {
            if (classfi[i] > max)
            {
                max = classfi[i];
                idx = i;
            }
        }

        //---------------------------------
        // E. Find the gesture
        //---------------------------------
        switch (idx)
        {
            case 0: cout << "ONE" << max <<endl;
                imshow("GUI", runImg);
                break;
            case 1: cout << "Palm" << max<<endl;
                imshow("GUI", walkImg);
                break;
            case 2: cout << "Fist" << max<<endl;
                imshow("GUI", stopImg);
                break;
            default:
                break;
        }

        // Output Images
        rectangle(img_display, matchLoc, Point(matchLoc.x + t_img.cols, matchLoc.y + t_img.rows), Scalar::all(0), 2, 8, 0);
        rectangle(results_img, matchLoc, Point(matchLoc.x + t_img.cols, matchLoc.y + t_img.rows), Scalar::all(0), 2, 8, 0);

        imshow(image_window, img_display);
        imshow(result_window, results_img);

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
	@param - result - reference to result Mat

	// Improvement need to be made on template matching to implement with
	pyramid building - sample down and detect
http://docs.opencv.org/doc/tutorials/imgproc/pyramids/pyramids.html
-------------------------------------------------*/
void tempMatchingOpenCV(Mat& s_img, Mat& t_img, Mat& result)
{
    // create result image
    int result_cols = s_img.cols - t_img.cols + 1;
    int result_rows = s_img.rows - t_img.rows + 1;

    result.create(result_rows, result_cols, CV_32FC1);

    // Perform template matching
    int match_method = 5;
    matchTemplate(s_img, t_img, result, match_method);

}



//Function that returns the maximum of 3 integers
// reference: LAB2
int myMax(int a, int b, int c) {

    int m = a;

    (void)((m < b) && (m = b));
    (void)((m < c) && (m = c));

    return m;

}



//Function that returns the minimum of 3 integers
// reference: LAB2
int myMin(int a, int b, int c) {

    int m = a;

    (void)((m > b) && (m = b));
    (void)((m > c) && (m = c));

    return m;

}


//Function that detects whether a pixel belongs to the skin based on RGB values
// reference: LAB2
void SkinDetect(Mat& src, Mat& dst) {
    //Surveys of skin color modeling and detection techniques:
    //Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
    //Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            //For each pixel, compute the average intensity of the 3 color channels
            Vec3b intensity = src.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)

            int B = intensity[0]; int G = intensity[1]; int R = intensity[2];

            if ((R > 95 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 15) && (abs(R - G) > 15) && (R > G) && (R > B)){
                dst.at<uchar>(i, j) = 255;
            }

        }

    }

}



//Function that does frame differencing between the current frame and the previous frame
// reference: LAB2
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst) {

    //For more information on operation with arrays: http://docs.opencv.org/modules/core/doc/operations_on_arrays.html
    //For more information on how to use background subtraction methods: http://docs.opencv.org/trunk/doc/tutorials/video/background_subtraction/background_subtraction.html
    Mat gs = dst.clone();

    absdiff(prev, curr, dst);
    cvtColor(dst, gs, CV_BGR2GRAY);

    dst = gs > 50;
    Vec3b intensity = dst.at<Vec3b>(100, 100);

}


//Function that accumulates the frame differences for a certain number of pairs of frames
// reference: LAB2
void myMotionEnergy(vector<Mat> mh, Mat& dst) {

    Mat mh0 = mh[0];
    Mat mh1 = mh[1];
    Mat mh2 = mh[2];

    for (int i = 0; i < dst.rows; i++){
        for (int j = 0; j < dst.cols; j++){
            if (mh0.at<uchar>(i, j) == 255 || mh1.at<uchar>(i, j) == 255 || mh2.at<uchar>(i, j) == 255){
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
}