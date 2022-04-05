#include <iostream>

#include <opencv2/opencv.hpp>

void DrawContours(cv::Mat& grayscale, cv::Mat& image)
{
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(grayscale, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++)
	{
		cv::Rect rect{ cv::boundingRect(contours[i]) };
		int area{ rect.area() };
		int imageArea{ image.size().area() };

		if (area >= 0.005 * imageArea && area <= 0.1 * imageArea)
		{
			cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 3);
		}
	}
}

void DetectEdges(cv::Mat& image, int lowerThreshold, int upperThreshold)
{
	cv::Mat dilationKernel = cv::getStructuringElement(cv::MORPH_DILATE, cv::Size(3, 3));

	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(image, image, cv::Size(3, 3), 3);
	cv::Canny(image, image, lowerThreshold, upperThreshold);
	cv::dilate(image, image, dilationKernel);
}

int main()
{
	int lowerThreshold{ 20 }, upperThreshold{ 70 };

	cv::namedWindow("Trackbars", (640, 200));
	cv::createTrackbar("Lower Threshold", "Trackbars", &lowerThreshold, 255);
	cv::createTrackbar("Upper Threshold", "Trackbars", &upperThreshold, 255);

	cv::VideoCapture camera;
	camera.open(0);

	while (camera.isOpened())
	{
		cv::Mat frame;
		camera >> frame;

		cv::Mat copy;
		frame.copyTo(copy);

		DetectEdges(copy, lowerThreshold, upperThreshold);
		DrawContours(copy, frame);

		imshow("Edges", frame);
		imshow("Copy", copy);

		cv::waitKey(1);
	}

	return 0;
}