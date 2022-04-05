#include <iostream>

#include <opencv2/opencv.hpp>

void DrawContours(const cv::Mat& image)
{
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);

		if (area > 1000.0)
		{
			cv::drawContours(image, contours, i, cv::Scalar(0, 255, 0), 3);
		}
	}
}

void DetectEdges(cv::Mat& image, int lowerThreshold, int upperThreshold)
{
	cv::Mat dilationKernel = cv::getStructuringElement(cv::MORPH_DILATE, cv::Size(3, 3));
	cv::Mat erosionKernel = cv::getStructuringElement(cv::MORPH_ERODE, cv::Size(3, 3));

	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(image, image, cv::Size(3, 3), 3);
	cv::Canny(image, image, lowerThreshold, upperThreshold);
	cv::dilate(image, image, dilationKernel);
	cv::erode(image, image, erosionKernel);
}

int main()
{
	/*std::string path = "apples.jpg";

	cv::Mat img = cv::imread(path);
	cv::Mat imgHSV, mask;

	int hmin = 0, smin = 150, vmin = 0;
	int hmax = 179, smax = 255, vmax = 255;

	cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

	cv::namedWindow("Trackbars", (640, 200));
	cv::createTrackbar("Hue Min", "Trackbars", &hmin, 179);
	cv::createTrackbar("Hue Max", "Trackbars", &hmax, 179);
	cv::createTrackbar("Sat Min", "Trackbars", &smin, 255);
	cv::createTrackbar("Sat Max", "Trackbars", &smax, 255);
	cv::createTrackbar("Val Min", "Trackbars", &vmin, 255);
	cv::createTrackbar("Val Max", "Trackbars", &vmax, 255);

	while (true) {

		cv::Scalar lower(hmin, smin, vmin);
		cv::Scalar upper(hmax, smax, vmax);
		cv::inRange(imgHSV, lower, upper, mask);

		cv::imshow("Image", img);
		cv::imshow("Image Mask", mask);
		cv::waitKey(1);
	}*/

	int lowerThreshold{ 0 }, upperThreshold{ 90 };

	cv::namedWindow("Trackbars", (640, 200));
	cv::createTrackbar("Lower Threshold", "Trackbars", &lowerThreshold, 255);
	cv::createTrackbar("Upper Threshold", "Trackbars", &upperThreshold, 255);

	cv::VideoCapture camera;
	camera.open(0);

	while (camera.isOpened())
	{
		cv::Mat frame;
		camera >> frame;

		DetectEdges(frame, lowerThreshold, upperThreshold);
		DrawContours(frame);

		imshow("Edges", frame);

		cv::waitKey(1);
	}

	return 0;
}