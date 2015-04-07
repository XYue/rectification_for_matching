#include "image_pair.hpp"

#include <iostream>

#include "opencv2/opencv.hpp"

namespace rect
{

	ImagePair::ImagePair()
	{
		Clear();
	}

	ImagePair::ImagePair( std::string yaml_flename )
	{
		Clear();

		if (ImportFromYAML(yaml_flename))
		{
			std::cout << "load camera from "<<yaml_flename<<" failed."<<std::endl;
			_left_image = Camera();
			_right_image = Camera();
		}
	}

	ImagePair::ImagePair( const Camera & left, const Camera & right )
	{
		Clear();

		_left_image = left;
		_right_image = right;
	}

	ImagePair::~ImagePair()
	{
	}

	int ImagePair::ImportFromYAML( std::string yaml_filename )
	{
		int ret = -1;

		do 
		{
			_left_image = Camera();
			_right_image = Camera();

			cv::FileStorage fs(yaml_filename, cv::FileStorage::READ);
			if(!fs.isOpened()) break;


			Camera left, right;

			fs["filename_l"] >> left._filename;
			fs["M_l"] >> left._CamMatrix;
			fs["D_l"] >> left._Distortion;
			fs["R_l"] >> left._R;
			fs["C_l"] >> left._C;
			if (_load_image_in_advanced)
			{
				left._image = cv::imread(left._filename,
					_load_image_in_gray_scale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
				if (left._image.empty()) break;
			}

			fs["filename_r"] >> right._filename;
			fs["M_r"] >> right._CamMatrix;
			fs["D_r"] >> right._Distortion;
			fs["R_r"] >> right._R;
			fs["C_r"] >> right._C;
			if (_load_image_in_advanced)
			{
				right._image = cv::imread(right._filename,
					_load_image_in_gray_scale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
				if (right._image.empty()) break;
			}


			// modified R due to the direction of R matrix by bundler
			cv::Mat tmp = left._R.row(0);
			tmp *= -1.;
			tmp.copyTo(left._R.row(0));
			tmp = left._R.row(2);
			tmp *= -1.;
			tmp.copyTo(left._R.row(2));

			tmp = right._R.row(0);
			tmp *= -1.;
			tmp.copyTo(right._R.row(0));
			tmp = right._R.row(2);
			tmp *= -1.;
			tmp.copyTo(right._R.row(2));


			_left_image = left;
			_right_image = right;

			
			if (image_pair_validation()) break;


			ret = 0;
		} while (0);
error0:

		return ret;
	}

	void ImagePair::Clear()
	{
		_left_image = Camera();
		_right_image = Camera();

		_load_image_in_gray_scale = false;
		_load_image_in_advanced = false;
	}

	int ImagePair::SaveRectifiedPair( std::string file_prefix )
	{
		int ret = -1;

		do 
		{
			cv::Mat left, right;
			if (_load_image_in_advanced)
			{
				left = _left_image._image;
				right = _right_image._image;
			} else {
				left = cv::imread(_left_image._filename, 
					_load_image_in_gray_scale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
				right = cv::imread(_right_image._filename, 
					_load_image_in_gray_scale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
			}
			if (left.empty() || right.empty()) break;


			cv::Mat R, T, R1, R2, P1, P2, Q;
			cv::Mat rectified_left, rectified_right;
			cv::Mat map11, map12, map21, map22;
			
			R = _right_image._R * _left_image._R.inv();
			T = _right_image._R * (_left_image._C-_right_image._C);
			cv::stereoRectify( _left_image._CamMatrix, _left_image._Distortion,
				_right_image._CamMatrix, _right_image._Distortion,
				left.size(), R, T, R1, R2, P1, P2, Q);

			initUndistortRectifyMap(_left_image._CamMatrix, _left_image._Distortion,
				R1, P1, left.size(), CV_32FC1, map11, map12);
			initUndistortRectifyMap(_right_image._CamMatrix, _right_image._Distortion,
				R2, P2, right.size(), CV_32FC1, map21, map22);
			remap(left, rectified_left, map11, map12, cv::INTER_LINEAR);
			remap(right, rectified_right, map21, map22, cv::INTER_LINEAR);

			imwrite(file_prefix + "_left.jpg", rectified_left);
			imwrite(file_prefix + "_right.jpg", rectified_right);


			double m[3] = {463,641,1};
			cv::Mat xy(3,1,R1.type(), m);
			cv::Mat K_n(3,3,R1.type());
			cv::Mat K(3,3,R1.type());
			_left_image._CamMatrix.convertTo(K, K.type());
			P1.col(0).copyTo(K_n.col(0));
			P1.col(1).copyTo(K_n.col(1));
			P1.col(2).copyTo(K_n.col(2));
			xy = K_n.inv() * xy;
			xy = R1.inv() * xy;
			xy /= xy.at<double>(2);
			xy = K * xy;
			std::cout<<K_n << std::endl;
			std::cout<<K << std::endl;
			std::cout<<xy<<std::endl;


			ret = 0;
		} while (0);
error0:

		return ret;
	}

	int ImagePair::image_pair_validation()
	{
		if ( (_load_image_in_advanced && 
			(_left_image._image.empty() || _right_image._image.empty())) ||
			_left_image._C.rows != 3 || _left_image._C.cols !=1 || 
			_right_image._C.rows != 3 || _right_image._C.cols !=1 ||
			_left_image._R.rows != 3 || _left_image._R.cols !=3 || 
			_right_image._R.rows != 3 || _right_image._R.cols !=3 ||
			_left_image._CamMatrix.rows != 3 || _left_image._CamMatrix.cols !=3 || 
			_right_image._CamMatrix.rows != 3 || _right_image._CamMatrix.cols !=3) return -1;
		else return 0;
	}

}