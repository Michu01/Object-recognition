#include <iostream>

#include <opencv2/opencv.hpp>

void detectImage(cv::Mat& image, cv::CascadeClassifier& cascade)
{
	std::vector<cv::Rect> faces;

	cascade.detectMultiScale(image, faces);

	for (const cv::Rect& face : faces)
	{
		cv::rectangle(image, face, cv::Scalar(0, 0, 255), 5);
	}
}

int main()
{
	try
	{
		cv::CascadeClassifier faceCascade;

		if (!faceCascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"))
		{
			throw std::exception("Empty face cascade");
		}

		cv::VideoCapture camera;
		camera.open(0);

		while (camera.isOpened())
		{
			cv::Mat frame;
			camera >> frame;

			cv::resize(frame, frame, cv::Size(), 1.5, 1.5);

			detectImage(frame, faceCascade);

			cv::imshow("OpenCV", frame);
			cv::waitKey(1);
		}
	}
	catch (std::exception e)
	{
		std::cout << e.what();
	}

	return 0;
}