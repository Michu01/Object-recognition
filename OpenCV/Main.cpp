#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

std::vector<std::string> loadNames()
{
	std::ifstream file("coco.names.txt");

	if (!file.is_open())
	{
		throw std::exception("Coco");
	}

	std::string line;
	std::vector<std::string> names;

	while (std::getline(file, line))
	{
		names.push_back(line);
	}

	return names;
}

cv::dnn::Net loadModel()
{
	cv::dnn::Net model{ cv::dnn::readNetFromTensorflow("model.pb", "config.txt") };

	if (model.empty())
	{
		throw std::exception("Empty model");
	}

	return model;
}

cv::VideoCapture loadCamera()
{
	cv::VideoCapture camera;
	
	if (!camera.open(0))
	{
		throw std::exception("Camera failed");
	}

	return camera;
}

void processFrame(cv::Mat& frame, cv::dnn::Net& model, const std::vector<std::string>& names)
{
	cv::Mat blob{ cv::dnn::blobFromImage(frame, 1.0 / 127.5, cv::Size(320, 320), cv::Scalar(127.5, 127.5, 127.5), true) };

	model.setInput(blob);

	cv::Mat output{ model.forward() };
	cv::Mat detection(output.size[2], output.size[3], CV_32F, output.ptr<float>());

	for (int i = 0; i < detection.rows; i++)
	{
		float confidence{ detection.at<float>(i, 2) };

		if (confidence > 0.5)
		{
			cv::Point tl{ static_cast<int>(detection.at<float>(i, 3) * frame.cols), static_cast<int>(detection.at<float>(i, 4) * frame.rows) };
			cv::Point bl{ static_cast<int>(detection.at<float>(i, 5) * frame.cols), static_cast<int>(detection.at<float>(i, 6) * frame.rows) };

			int nameIndex{ static_cast<int>(detection.at<float>(i, 1)) };

			cv::rectangle(frame, tl, bl, cv::Scalar(255, 255, 255), 2);
			cv::putText(frame, names[nameIndex - 1], tl - cv::Point{ 0, 5 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
		}
	}
}

int main()
{
	try 
	{
		std::vector<std::string> names{ loadNames() };

		cv::dnn::Net model{ loadModel() };

		cv::VideoCapture camera{ loadCamera() };

		while (camera.isOpened())
		{
			cv::Mat frame{};
			camera >> frame;

			processFrame(frame, model, names);

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